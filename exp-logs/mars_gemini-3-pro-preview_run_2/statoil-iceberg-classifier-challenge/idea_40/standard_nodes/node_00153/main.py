import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# 1. Configuration & Patching
# Patch NUM_EPOCHS in config before importing trainer to ensure fast execution
# while maintaining the integrity of the provided library files.
import library.config

library.config.NUM_EPOCHS = 50

from library.config import SEED, NUM_FOLDS, SUBMISSION_DIR
from library.utils import set_seed
from library.data import process_and_cache_data
from library.trainer import run_fold


def main():
    # Set global seed for reproducibility
    set_seed(SEED)

    print("Initializing QC-WBN Pipeline...")

    # 2. Load Data & Metadata
    # We load data here to replicate the stratified split for aggregation and analysis
    data = process_and_cache_data(load_cached_data=True)
    train_images = data["train_images"]
    train_angles = data["train_angles"]
    train_labels = data["train_labels"]
    # train_ids = data["train_ids"]

    # Prepare storage for results
    oof_preds_accum = []
    oof_targets_accum = []
    oof_indices_accum = []
    test_preds_accum = []
    fold_test_ids = None

    # 3. Stratified K-Fold Training Loop
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    print(f"Starting {NUM_FOLDS}-Fold Cross-Validation...")

    # Iterate through folds to match the logic in library/data.py
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_images, train_labels)
    ):
        print(f"\n--- Fold {fold_idx} ---")

        # Execute training for the current fold
        # debug=False ensures we use the full dataset
        result = run_fold(fold_idx, debug=False)

        # Store OOF predictions and targets
        # result['val_preds'] are probabilities for the validation set of this fold
        oof_preds_accum.append(result["val_preds"])
        oof_targets_accum.append(result["val_labels"])
        oof_indices_accum.append(val_idx)

        # Store Test predictions
        test_preds_accum.append(result["test_preds"])

        # Capture test IDs from the first fold (they are constant)
        if fold_test_ids is None:
            fold_test_ids = result["test_ids"]

    # 4. Evaluation & Aggregation
    # Concatenate all OOF results to form a global validation set
    all_oof_preds = np.concatenate(oof_preds_accum)
    all_oof_targets = np.concatenate(oof_targets_accum)
    all_oof_indices = np.concatenate(oof_indices_accum)

    # Compute Final Validation Metric (Log Loss)
    final_metric = log_loss(all_oof_targets, all_oof_preds)
    print(f"\nFinal Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis Report")
    print("=======================")

    # Retrieve features for the validation samples using the stored indices
    # This ensures we are correlating errors with the correct input data
    val_angles = train_angles[all_oof_indices]
    val_images = train_images[all_oof_indices]

    # Calculate absolute error magnitude
    errors = np.abs(all_oof_targets - all_oof_preds)

    # Compute derived features for analysis
    # Mean intensity of Band 1 and Band 2
    b1_mean = np.mean(val_images[..., 0], axis=(1, 2))
    b2_mean = np.mean(val_images[..., 1], axis=(1, 2))

    # Create analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": val_angles,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
            "target": all_oof_targets,
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = 0.15744295919935183

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is below threshold ({threshold}). Generating submission..."
        )

        # Ensemble: Average predictions across all folds
        avg_test_preds = np.mean(test_preds_accum, axis=0)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"id": fold_test_ids, "is_iceberg": avg_test_preds}
        )

        # Save to file
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
