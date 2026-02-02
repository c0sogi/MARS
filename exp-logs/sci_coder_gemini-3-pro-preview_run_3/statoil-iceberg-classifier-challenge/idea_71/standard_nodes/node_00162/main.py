import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
import library.config as config
import library.train as train_lib
from library.utils import set_seed, setup_logger
from library.data_loader import process_data


def main():
    # 1. Setup
    # Set seed for reproducibility
    set_seed(config.SEED)

    # Setup logger
    log_path = os.path.join(config.WORKING_DIR, "runfile.log")
    logger = setup_logger(log_path)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # Optimize runtime for baseline execution
    # Reducing epochs to 50 to ensure completion within time limit while allowing convergence
    train_lib.NUM_EPOCHS = 50
    logger.info(f"Training for {train_lib.NUM_EPOCHS} epochs per fold.")

    # 2. Data Loading
    # Load cached data if available, otherwise process from scratch
    logger.info("Loading data...")
    X_full, y_full, angle_full, X_test, ids_test, angle_test = process_data(
        load_cached_data=True
    )

    logger.info(f"Train set shape: {X_full.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # 3. Cross-Validation Training Loop
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Arrays to store Out-Of-Fold (OOF) predictions and accumulated Test predictions
    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros((len(X_test), 1))

    logger.info("\nStarting Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Run training for the current fold
        # run_training_fold handles the splitting internally using the same StratifiedKFold logic
        val_preds, val_targets, fold_test_preds, best_loss = (
            train_lib.run_training_fold(
                fold_idx, X_full, y_full, angle_full, X_test, angle_test, device, logger
            )
        )

        # Store OOF predictions
        # val_idx here matches the validation set used inside run_training_fold
        oof_preds[val_idx] = val_preds.flatten()

        # Accumulate Test predictions (for averaging later)
        test_preds_accum += fold_test_preds

        logger.info(f"Fold {fold_idx + 1} completed. Best Loss: {best_loss:.6f}")

    # 4. Metric Calculation
    # Calculate Log Loss on the full OOF predictions
    final_metric = log_loss(y_full, oof_preds)

    # Print the exact required string
    print(f"Final Validation Metric: {final_metric}")
    logger.info(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("\n--- Failure Analysis ---")

    # Calculate absolute error per sample
    errors = np.abs(y_full - oof_preds)

    # Extract features for correlation analysis
    # X_full shape: (N, 3, 75, 75). Channel 0: HH (Band 1), Channel 1: HV (Band 2)
    b1_mean = np.mean(X_full[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_full[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_full[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_full[:, 1, :, :], axis=(1, 2))

    # Handle missing incidence angles for analysis (impute with median)
    ang_median = np.nanmedian(angle_full)
    angles_filled = np.nan_to_num(angle_full, nan=ang_median)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_filled,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Calculate correlation with error
    corrs = analysis_df.corr()["error"].drop("error")

    print("Correlation between Error and Features:")
    print(corrs)
    logger.info("Correlation between Error and Features:")
    logger.info(corrs)

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        logger.info(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Average test predictions across folds
        avg_test_preds = test_preds_accum / config.NUM_FOLDS

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"id": ids_test, "is_iceberg": avg_test_preds.flatten()}
        )

        # Save to file
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)

        print(f"Submission saved to {sub_path}")
        logger.info(f"Submission saved to {sub_path}")
    else:
        logger.info(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
