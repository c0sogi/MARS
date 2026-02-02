import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library components
from library.config import (
    DEVICE,
    NUM_FOLDS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    NUM_WORKERS,
    SEED,
    IDEA_ID,
)
from library.utils import set_seed, setup_logger, save_checkpoint
from library.data import IcebergDataset, load_and_process_json, get_transforms
from library.model import CDICNN
from library.train import train_one_epoch, validate, inference

# =============================================================================
# CONFIGURATION FOR BASELINE
# =============================================================================
# Reduced epochs for fast baseline execution as requested
LOCAL_NUM_EPOCHS = 30
PATIENCE = 10
TARGET_METRIC_THRESHOLD = 0.17174082291273365


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    logger = setup_logger(os.path.join(CHECKPOINT_DIR, "runfile.log"))
    logger.info(f"Starting Fast Baseline Execution for {IDEA_ID}")

    # 2. Load Data
    logger.info("Loading and processing data...")
    # Load full training data
    X_full, ang_full, y_full, ids_train = load_and_process_json(
        TRAIN_JSON_PATH, "train_full", load_cached=True
    )

    # Load test data
    X_test, ang_test, _, ids_test = load_and_process_json(
        TEST_JSON_PATH, "test", load_cached=True
    )

    # Impute missing angles in Test set using global median of training data
    valid_train_angles = ang_full[~np.isnan(ang_full)]
    global_median = (
        np.median(valid_train_angles) if len(valid_train_angles) > 0 else 0.0
    )
    ang_test[np.isnan(ang_test)] = global_median

    # Prepare Test Loader (for later use)
    test_dataset = IcebergDataset(
        X_test, ang_test, None, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Store OOF predictions
    oof_preds = np.zeros(len(X_full))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"\n========== Fold {fold} ==========")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        ang_train_fold, ang_val_fold = ang_full[train_idx], ang_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Impute Angles (Compute median on training fold only)
        valid_fold_angles = ang_train_fold[~np.isnan(ang_train_fold)]
        fold_median = (
            np.median(valid_fold_angles) if len(valid_fold_angles) > 0 else 0.0
        )

        ang_train_fold[np.isnan(ang_train_fold)] = fold_median
        ang_val_fold[np.isnan(ang_val_fold)] = fold_median

        # Create Datasets and Loaders
        train_ds = IcebergDataset(
            X_train_fold,
            ang_train_fold,
            y_train_fold,
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_val_fold, ang_val_fold, y_val_fold, transform=get_transforms("val")
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = CDICNN().to(DEVICE)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0

        # Training Loop
        for epoch in range(1, LOCAL_NUM_EPOCHS + 1):
            _ = train_one_epoch(
                train_loader, model, criterion, optimizer, DEVICE, epoch, logger
            )
            val_loss = validate(val_loader, model, criterion, DEVICE, logger)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_loss": val_loss,
                    },
                    is_best=True,
                    checkpoint_dir=CHECKPOINT_DIR,
                    fold_idx=fold,
                )
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model for OOF prediction
        best_model_path = os.path.join(CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        # Generate OOF predictions
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.extend(probs)

        oof_preds[val_idx] = np.array(fold_preds)

    # 4. Validation Assessment
    final_log_loss = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_full - oof_preds)

    # Calculate feature stats for correlation
    # Band 1 Mean (Channel 0)
    b1_mean = np.mean(X_full[:, 0, :, :], axis=(1, 2))
    # Band 2 Mean (Channel 1)
    b2_mean = np.mean(X_full[:, 1, :, :], axis=(1, 2))

    # Create DataFrame for analysis
    fa_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_full,  # Note: this contains NaNs, correlation will ignore or we fill
            "b1_mean": b1_mean,
            "b2_mean": b2_mean,
        }
    )

    # Fill NaNs in inc_angle for correlation calculation
    fa_df["inc_angle"] = fa_df["inc_angle"].fillna(fa_df["inc_angle"].median())

    # Calculate correlation
    correlations = fa_df.corr()["error"].drop("error")
    print("Failure Analysis - Correlation with Error:")
    print(correlations)

    # 6. Conditional Submission
    if final_log_loss < TARGET_METRIC_THRESHOLD:
        logger.info(
            f"Validation metric {final_log_loss} is better than threshold {TARGET_METRIC_THRESHOLD}. Generating submission..."
        )

        fold_test_preds = np.zeros((len(X_test), NUM_FOLDS))

        for fold in range(NUM_FOLDS):
            logger.info(f"Inference for Fold {fold}...")
            model = CDICNN().to(DEVICE)
            best_model_path = os.path.join(
                CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
            )
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["state_dict"])

            preds = inference(test_loader, model, DEVICE)
            fold_test_preds[:, fold] = preds

        # Average predictions
        avg_preds = np.mean(fold_test_preds, axis=1)

        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")
    else:
        logger.info(
            f"Validation metric {final_log_loss} did not meet threshold {TARGET_METRIC_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
