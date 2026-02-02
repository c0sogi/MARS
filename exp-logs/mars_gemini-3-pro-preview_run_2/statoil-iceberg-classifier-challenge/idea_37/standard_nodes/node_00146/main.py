import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided libraries
from library.config import DEVICE, SUBMISSION_DIR, METADATA_DIR
from library.utils import set_seed, get_logger
from library.data_loader import process_and_cache_data, IcebergDataset
from library.train_eval import train_fold, predict

# Initialize Logger
logger = get_logger("runfile")


def analyze_failures_oof(y_true, y_pred, angles):
    """
    Performs failure analysis on the OOF predictions.
    Calculates correlation between error magnitude and incidence angle.
    """
    # Calculate Error Magnitude (Absolute Error)
    errors = np.abs(y_true - y_pred)

    # Calculate Correlation with Incidence Angle
    if len(errors) > 1:
        corr, p_value = pearsonr(errors, angles)
    else:
        corr = 0.0

    print(f"Correlation between Error Magnitude and Incidence Angle: {corr:.6f}")
    return corr


def main():
    # 1. Setup
    set_seed(42)
    logger.info("Starting runfile execution...")

    # 2. Data Loading (Load All Data for K-Fold)
    logger.info("Loading processed data...")
    data = process_and_cache_data(load_cached_data=True)

    X = data["train_images"]
    angles = data["train_angles"]
    y = data["train_labels"]

    X_test = data["test_images"]
    angles_test = data["test_angles"]
    test_ids = data["test_ids"]

    # Prepare Test Loader
    test_dataset = IcebergDataset(X_test, angles_test, labels=None, transform=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    # 3. Stratified K-Fold Ensemble (Cite Lesson 00052, 00143)
    # Lesson 00052: Prioritize K-Fold Cross-Validation Ensembling over Fixed-Split.
    # Lesson 00143: Ensemble Discrepancy Principle - use ensemble for final comparison.
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(test_ids))

    logger.info(f"Starting {n_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Fold {fold}/{n_folds}")

        # Create Fold Datasets
        train_ds = IcebergDataset(
            X[train_idx], angles[train_idx], y[train_idx], transform=True
        )
        val_ds = IcebergDataset(
            X[val_idx], angles[val_idx], y[val_idx], transform=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=32, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=32, shuffle=False, num_workers=2
        )

        # Train Model (Returns best model for this fold)
        model = train_fold(
            fold_idx=fold,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,  # Increased epochs slightly, relying on early stopping
            patience=10,
            device=DEVICE,
        )

        # OOF Predictions
        fold_oof_preds = predict(model, val_loader, DEVICE)
        oof_preds[val_idx] = fold_oof_preds

        # Test Predictions
        fold_test_preds = predict(model, test_loader, DEVICE)
        test_preds_accum += fold_test_preds

    # Average Test Predictions
    avg_test_preds = test_preds_accum / n_folds

    # 4. Validation Metric (Global OOF Log Loss)
    final_metric = log_loss(y, oof_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis (On OOF Data)
    logger.info("Performing failure analysis on OOF predictions...")
    analyze_failures_oof(y, oof_preds, angles)

    # 6. Submission Logic
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        # Create Submission DataFrame
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})

        # Save
        df_sub.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.info(
            f"Validation metric {final_metric} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
