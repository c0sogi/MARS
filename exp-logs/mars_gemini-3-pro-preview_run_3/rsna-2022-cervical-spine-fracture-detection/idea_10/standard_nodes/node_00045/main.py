import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config, seed_everything
from library.utils import load_and_preprocess_scan
from library.dataset import FractureDataset, get_transforms
from library.model import CervicalFractureNet
from library.loss import HierarchicalCompoundLoss
from library.engine import fit


def calculate_weighted_metric(y_true_c, y_pred_c, y_true_p, y_pred_p):
    """
    Calculates the competition metric:
    Weighted Log Loss where Patient Overall is weighted 1 and each Vertebra is 1/7.
    This corresponds to the implicit weighting in the loss function:
    Total Loss = Mean(Vertebral_Losses) + Patient_Loss.
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred_c = np.clip(y_pred_c, epsilon, 1 - epsilon)
    y_pred_p = np.clip(y_pred_p, epsilon, 1 - epsilon)

    # Calculate Log Loss for each vertebra column (C1-C7)
    losses_c = []
    for i in range(7):
        # Handle cases where a class might be all 0 or all 1 in validation batch
        labels = [0, 1]
        l = log_loss(y_true_c[:, i], y_pred_c[:, i], labels=labels)
        losses_c.append(l)

    # Calculate Log Loss for patient overall
    loss_p = log_loss(y_true_p, y_pred_p, labels=[0, 1])

    # Metric = Mean(Vertebral Losses) + Patient Loss
    # This effectively weights the patient_overall label 7 times higher than a single vertebra label
    metric = np.mean(losses_c) + loss_p
    return metric


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Pre-caching Data
    # To ensure training speed, we pre-process the training and validation scans.
    # This converts DICOMs to 2.5D .npy stacks.
    print("Pre-caching Training and Validation data...")
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine unique study UIDs from train and val
    unique_uids = pd.concat(
        [train_meta["StudyInstanceUID"], val_meta["StudyInstanceUID"]]
    ).unique()

    for uid in unique_uids:
        # load_and_preprocess_scan handles checking cache and generating if missing
        # We use Config.TRAIN_IMAGES_DIR because both train and val splits come from the training set images
        load_and_preprocess_scan(uid, Config.TRAIN_IMAGES_DIR, load_cached_data=True)

    # 3. Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_dataset = FractureDataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("train"),
    )
    val_dataset = FractureDataset(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for BatchNorm stability
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model, Loss, Optimizer
    print("Initializing Model...")
    model = CervicalFractureNet().to(device)

    criterion = HierarchicalCompoundLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.EPOCHS * Config.T_MAX_MULT), eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    print("Starting Training...")
    # fit() handles the training loop, validation, and saving the best model
    model = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        epochs=Config.EPOCHS,
        patience=5,
    )

    # 6. Validation Assessment
    print("Evaluating on Validation Set for Final Metric...")
    model.eval()

    val_preds_c = []
    val_targets_c = []
    val_targets_p = []
    val_uids = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            targs = batch["targets"].to(device)
            pat_targs = batch["patient_target"].to(device)
            uids = batch["study_uid"]

            # Forward pass
            logits = model(imgs)  # Shape: (Batch, 7)
            probs = torch.sigmoid(logits)

            val_preds_c.append(probs.cpu().numpy())
            val_targets_c.append(targs.cpu().numpy())
            val_targets_p.append(pat_targs.cpu().numpy())
            val_uids.extend(uids)

    val_preds_c = np.concatenate(val_preds_c)
    val_targets_c = np.concatenate(val_targets_c)
    val_targets_p = np.concatenate(val_targets_p)

    # Derive patient_overall prediction as the max probability of C1-C7
    val_preds_p = np.max(val_preds_c, axis=1)

    final_metric = calculate_weighted_metric(
        val_targets_c, val_preds_c, val_targets_p, val_preds_p
    )

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error per sample to correlate with metadata
    sample_errors = []
    epsilon = 1e-15
    val_preds_c_clip = np.clip(val_preds_c, epsilon, 1 - epsilon)
    val_preds_p_clip = np.clip(val_preds_p, epsilon, 1 - epsilon)

    for i in range(len(val_uids)):
        # Calculate loss contribution for this sample
        # Mean log loss of C1-C7
        loss_c = log_loss(
            val_targets_c[i],
            val_preds_c_clip[i],
            labels=[0, 1],
        )
        # Log loss of patient
        loss_p = log_loss([val_targets_p[i]], [val_preds_p_clip[i]], labels=[0, 1])
        # Total error
        sample_errors.append(loss_c + loss_p)

    # Get metadata feature: Number of slices (Scan Depth)
    # We count files in the directory as a proxy for scan complexity/depth
    num_slices = []
    for uid in val_uids:
        path = os.path.join(Config.TRAIN_IMAGES_DIR, uid)
        try:
            # Fast way to count files
            cnt = sum(1 for _ in os.scandir(path))
        except:
            cnt = 0
        num_slices.append(cnt)

    # Calculate Correlation
    if len(sample_errors) > 1:
        corr = np.corrcoef(sample_errors, num_slices)[0, 1]
    else:
        corr = 0.0

    print(f"Correlation between Error and Num_Slices: {corr}")

    # 8. Submission Generation
    TARGET_METRIC = 0.12231192492082398

    if final_metric < TARGET_METRIC:
        print(f"Metric {final_metric} < {TARGET_METRIC}. Generating submission...")

        # Initialize Test Dataset
        test_dataset = FractureDataset(
            Config.TEST_METADATA_PATH,
            Config.TEST_IMAGES_DIR,
            transform=get_transforms("test"),
            mode="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_preds = []
        all_uids = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(device)
                uids = batch["study_uid"]

                logits = model(imgs)
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())
                all_uids.extend(uids)

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)

            row_ids = []
            fractured = []

            for i, uid in enumerate(all_uids):
                preds = all_preds[i]  # 7 probabilities (C1-C7)
                p_patient = np.max(preds)  # Derived patient overall

                # Append C1-C7 rows
                for c_idx in range(7):
                    row_ids.append(f"{uid}_C{c_idx+1}")
                    fractured.append(preds[c_idx])

                # Append Patient Overall row
                row_ids.append(f"{uid}_patient_overall")
                fractured.append(p_patient)

            submission_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured})
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print("No test predictions generated.")

    else:
        print(f"Metric {final_metric} >= {TARGET_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()
