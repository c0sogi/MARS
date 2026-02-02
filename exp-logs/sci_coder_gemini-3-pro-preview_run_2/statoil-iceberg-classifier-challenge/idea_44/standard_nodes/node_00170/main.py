import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Import provided library modules
from library import config, utils, data_loader, model, train


def run_pipeline():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # 2. Training Phase
    # Run the 5-fold CV training. This saves models to config.WORK_DIR
    # We use debug_size=None to train on the full dataset for maximum performance.
    print("Starting Training Phase...")
    train.run_training(debug_size=None)

    # 3. Validation Phase
    print("\nStarting Validation Phase...")

    # Load validation metadata to identify hold-out samples
    val_meta_path = config.VAL_META
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    df_val_meta = pd.read_csv(val_meta_path)
    val_ids_set = set(df_val_meta["id"].values)

    # Load cached preprocessed data
    # Training has just completed, so cache exists.
    train_data, test_data, stats = data_loader.process_and_cache_data(
        load_cached_data=True
    )

    # Filter training data to get only the validation set
    all_ids = train_data["ids"]
    mask = np.isin(all_ids, list(val_ids_set))

    val_images = train_data["images"][mask]
    val_angles = train_data["angles"][mask]
    val_labels = train_data["labels"][mask]
    val_ids = train_data["ids"][mask]

    print(f"Validation set size: {len(val_ids)}")

    # OOF Inference on Validation Set
    # Since we trained on the full dataset (Lesson 00165), we must use Out-Of-Fold predictions
    # to evaluate the hold-out set without leakage.
    print("Generating OOF predictions for validation...")

    oof_preds_dict = {}

    # We need the full training data to replicate the split
    images_all = train_data["images"]
    labels_all = train_data["labels"]
    ids_all = train_data["ids"]
    angles_all = train_data["angles"]

    # Replicate the split used in training
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(images_all, labels_all)):
        model_path = os.path.join(config.WORK_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        # Create val dataset for this fold
        fold_val_dataset = data_loader.IcebergDataset(
            images=images_all[val_idx],
            angles=angles_all[val_idx],
            stats=stats,
            labels=labels_all[val_idx],
            ids=ids_all[val_idx],
            transform=False,
        )
        fold_val_loader = DataLoader(
            fold_val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        # Predict
        net = model.InputAnchoredWideBodyNet().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint)
        net.eval()

        with torch.no_grad():
            for imgs, angs, _, batch_ids in fold_val_loader:
                imgs, angs = imgs.to(device), angs.to(device)
                outputs = net(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                for i, pid in enumerate(batch_ids):
                    oof_preds_dict[pid] = probs[i]

    # Extract predictions for the specific hold-out validation IDs
    avg_preds = []
    for vid in val_ids:
        if vid in oof_preds_dict:
            avg_preds.append(oof_preds_dict[vid])
        else:
            # Fallback (should not happen with full dataset training)
            avg_preds.append(0.5)

    avg_preds = np.array(avg_preds)

    # Calculate Metric (Log Loss)
    # Clip to prevent log(0)
    avg_preds_clipped = np.clip(avg_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_labels, avg_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nFailure Analysis...")
    errors = np.abs(val_labels - avg_preds)

    # Compute simple image stats for correlation analysis
    # val_images is (N, 75, 75, 3). Band 0: HH, Band 1: HV
    b1_mean = np.mean(val_images[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(val_images[:, :, :, 1], axis=(1, 2))

    fa_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": val_angles,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
        }
    )

    correlations = fa_df.corr()["error"].drop("error")
    print("Correlations with Error Magnitude:")
    print(correlations)

    # 5. Submission
    THRESHOLD = 1.0

    if final_metric < THRESHOLD:
        print("\nMetric meets threshold. Generating submission...")

        # Prepare Test Loader
        test_dataset = data_loader.IcebergDataset(
            images=test_data["images"],
            angles=test_data["angles"],
            stats=stats,
            labels=None,
            ids=test_data["ids"],
            transform=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        test_fold_preds = []

        # Inference on Test Set
        for fold in range(config.NUM_FOLDS):
            model_path = os.path.join(config.WORK_DIR, f"model_fold_{fold}.pth")
            if not os.path.exists(model_path):
                continue

            net = model.InputAnchoredWideBodyNet().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint)
            net.eval()

            preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = net(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    preds.extend(probs)

            test_fold_preds.append(np.array(preds))

        # Ensemble Average
        avg_test_preds = np.mean(test_fold_preds, axis=0)

        # Save Submission
        sub_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": avg_test_preds})

        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
