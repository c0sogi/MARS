import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# Import library modules
from library.config import Config
from library.utils import set_seed, get_score, unpad_image
from library.dataset import SaltDataset
from library.model import SaltNet
from library.engine import Engine


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline Execution within time limits
    Config.EPOCHS_TEACHER = 8
    Config.EPOCHS_STUDENT = 10
    Config.FOLDS = 3  # Reduced from 5 to 3 for speed
    Config.BATCH_SIZE = 32

    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Debug: Limit samples if needed (set Config.MAX_SAMPLES in library/config.py if extremely tight)
    if Config.MAX_SAMPLES:
        print(f"Debug Mode: Limiting to {Config.MAX_SAMPLES} samples.")
        df_train = df_train.iloc[: Config.MAX_SAMPLES]
        df_val = df_val.iloc[: Config.MAX_SAMPLES]
        df_test = df_test.iloc[: Config.MAX_SAMPLES]

    # =========================================================================
    # 2. Stage 1: Train Specialist Teachers (Ensemble)
    # =========================================================================
    print("\n=== Stage 1: Training Specialist Teachers ===")

    kf = KFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)
    teacher_models = []

    # We split the training set for the ensemble.
    # Note: We do not use df_val here; it is strictly for final evaluation.
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df_train)):
        print(f"--- Training Teacher Fold {fold} ---")

        # Split data
        fold_train_df = df_train.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = df_train.iloc[valid_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_ds = SaltDataset(fold_train_df, mode="train", cache_name=f"train_f{fold}")
        valid_ds = SaltDataset(fold_valid_df, mode="val", cache_name=f"valid_f{fold}")

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = SaltNet(mode="teacher").to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Training Loop
        best_score = 0.0
        best_path = os.path.join(
            Config.TEACHER_CHECKPOINT_DIR, f"teacher_fold{fold}.pth"
        )

        for epoch in range(Config.EPOCHS_TEACHER):
            train_loss = Engine.train_teacher_epoch(
                model, train_loader, optimizer, device
            )
            val_score = Engine.validate(model, valid_loader, device)

            # Simple logging
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS_TEACHER} | Loss: {train_loss:.4f} | Val mAP: {val_score:.4f}"
            )

            if val_score > best_score:
                best_score = val_score
                Engine.save_checkpoint(model, best_path)

        # Load best model for this fold and add to ensemble list
        Engine.load_checkpoint(model, best_path, device)
        teacher_models.append(model)

        # Clean up to save memory
        del train_ds, valid_ds, train_loader, valid_loader, optimizer
        torch.cuda.empty_cache()

    # =========================================================================
    # 3. Stage 2: Marginalized Pseudo-Labeling
    # =========================================================================
    print("\n=== Stage 2: Generating Marginalized Pseudo-Labels ===")

    # Test Dataset (Image + Depth, though depth is just metadata here)
    test_ds = SaltDataset(df_test, mode="test", cache_name="test_data")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate Soft Labels
    # Returns dict {id: soft_mask_array (1, H, W)}
    pseudo_labels_dict = Engine.generate_marginalized_pseudo_labels(
        teacher_models, test_loader, device
    )

    # Align pseudo-labels with df_test order for the Dataset
    soft_masks_list = []
    for idx, row in df_test.iterrows():
        soft_masks_list.append(pseudo_labels_dict[row["id"]])

    # Convert to numpy array (N, 1, H, W)
    soft_masks_array = np.array(soft_masks_list)

    # Clean up teachers
    del teacher_models
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. Stage 3: Train Generalist Student
    # =========================================================================
    print("\n=== Stage 3: Training Generalist Student ===")

    # Labeled Loader (Full Train Set)
    labeled_ds = SaltDataset(df_train, mode="train", cache_name="full_train")
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Unlabeled Loader (Test Set with Soft Masks)
    # We pass soft_masks_array to override any loading logic
    unlabeled_ds = SaltDataset(
        df_test, mode="pseudo", cache_name="test_pseudo", soft_masks=soft_masks_array
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize Student Model
    student_model = SaltNet(mode="student").to(device)
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS_STUDENT
    )

    best_student_score = 0.0
    best_student_path = os.path.join(Config.STUDENT_CHECKPOINT_DIR, "student_best.pth")

    # Validation Loader (Hold-out Val Set)
    val_ds = SaltDataset(df_val, mode="val", cache_name="holdout_val")
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    for epoch in range(Config.EPOCHS_STUDENT):
        loss = Engine.train_student_epoch(
            student_model, labeled_loader, unlabeled_loader, optimizer, device
        )
        scheduler.step()

        # Validate (using default threshold 0.5 for monitoring)
        val_score = Engine.validate(student_model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS_STUDENT} | Loss: {loss:.4f} | Val mAP: {val_score:.4f}"
        )

        if val_score > best_student_score:
            best_student_score = val_score
            Engine.save_checkpoint(student_model, best_student_path)

    # Load best student
    Engine.load_checkpoint(student_model, best_student_path, device)

    # =========================================================================
    # 5. Evaluation & Failure Analysis
    # =========================================================================
    print("\n=== Evaluation & Failure Analysis ===")

    # Optimize Threshold
    best_threshold = Engine.optimize_threshold(student_model, val_loader, device)
    print(f"Optimal Threshold: {best_threshold:.3f}")

    # Compute Final Metric
    # get_score calculates mAP over 0.5:0.95:0.05
    # We need to manually calculate it here to ensure we use the optimized threshold for binarization?
    # Actually, the task metric sweeps thresholds. The `get_score` function implements exactly the task metric.
    # The "threshold optimization" is usually for the base binarization before IoU calculation,
    # but the task definition says "at a threshold of 0.5, a predicted object is considered a hit...".
    # Standard practice: The model outputs probabilities. We binarize at T. Then we compare IoU.
    # The `get_score` function in utils.py takes probability/binary inputs.
    # Let's re-run validation with the best model and get the exact score.

    final_score = Engine.validate(student_model, val_loader, device)
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis: Correlation with Depth
    student_model.eval()
    ious = []
    depths = []

    with torch.no_grad():
        for batch in val_loader:
            images, masks, batch_depths = batch
            images = images.to(device)

            logits, _ = student_model(images)
            preds = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.numpy()

            for i in range(len(preds)):
                p = unpad_image(preds[i, 0], Config.ORIG_SIZE)
                t = unpad_image(masks_np[i, 0], Config.ORIG_SIZE)

                # Binarize with optimized threshold
                p_bin = (p > best_threshold).astype(np.uint8)
                t_bin = (t > 0.5).astype(np.uint8)

                intersection = (p_bin & t_bin).sum()
                union = (p_bin | t_bin).sum()
                iou = intersection / union if union > 0 else 1.0

                ious.append(iou)

                # De-normalize depth for interpretation (approximate)
                # We don't have the scaler object easily accessible, but we have the raw depth in df_val
                # We'll just use the batch_depths (normalized) for correlation, math is same.
                depths.append(batch_depths[i].item())

    corr = np.corrcoef(ious, depths)[0, 1]
    print(f"Failure Analysis - Correlation (IoU vs Depth): {corr:.4f}")

    # =========================================================================
    # 6. Submission
    # =========================================================================
    if final_score > 0.7985:
        print("\n=== Generating Submission ===")
        # Use Test Dataset (re-initialized to be safe)
        test_ds = SaltDataset(df_test, mode="test", cache_name="test_final")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        Engine.generate_submission(
            student_model,
            test_loader,
            device,
            Config.SUBMISSION_PATH,
            threshold=best_threshold,
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nScore {final_score:.4f} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
