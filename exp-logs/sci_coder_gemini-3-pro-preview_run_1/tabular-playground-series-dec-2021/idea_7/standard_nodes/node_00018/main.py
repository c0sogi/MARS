import pandas as pd
import numpy as np
import os

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TARGET_COL,
    ID_COL,
    SUBMISSION_PATH,
)
from library.utils import set_seed, verify_gpu, save_submission
from library.data_processing import FeatureEngineer
from library.stacking_trainer import StackingManager


def main():
    # Set seed for reproducibility across all operations
    set_seed(42)

    # Verify GPU availability to ensure fast training
    verify_gpu()

    print("Loading data...")
    # Load datasets from the metadata directory
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Extract validation targets for scoring
    val_y = val_df[TARGET_COL].values

    # Extract Test IDs for submission file generation
    test_ids = test_df[ID_COL].values

    print("Processing data...")
    # Initialize Feature Engineer
    fe = FeatureEngineer()

    # Apply feature engineering (Reverse One-Hot, Geometric Features)
    train_df = fe.process(train_df)
    val_df = fe.process(val_df)
    test_df = fe.process(test_df)

    # Prepare Combined Evaluation Set (Validation + Test)
    # We combine them to run inference in a single pass through the StackingManager.
    # This avoids the significant overhead of retraining the ensemble models twice.
    # We drop the target from val_df to match the test_df schema for the 'test' argument.
    val_df_no_target = val_df.drop(columns=[TARGET_COL], errors="ignore")

    # Concatenate along rows
    eval_df = pd.concat([val_df_no_target, test_df], axis=0, ignore_index=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Eval (Val + Test) shape: {eval_df.shape}")

    # Initialize Stacking Manager
    manager = StackingManager()

    # --- Level 0: Base Learners ---
    # Train Base Learners and get predictions for Train (OOF) and Eval (Averaged)
    # We set load_cached_preds=False to force execution on our custom eval_df
    print("Running Level 0...")
    oof_preds, eval_l0_preds = manager.run_level_0(
        train_df, eval_df, load_cached_preds=False
    )

    # --- Level 1: Meta Learners ---
    # Train Meta Learners and get final class predictions for Eval
    print("Running Level 1...")
    eval_final_preds = manager.run_level_1(oof_preds, eval_l0_preds, train_df, eval_df)

    # --- Split Predictions ---
    # Separate the combined results back into validation and test sets
    n_val = len(val_df)
    val_preds = eval_final_preds[:n_val]
    test_preds = eval_final_preds[n_val:]

    # --- Validation Metric ---
    # Calculate multi-class accuracy
    acc = np.mean(val_preds == val_y)
    # Print full precision as required
    print(f"Final Validation Metric: {acc}")

    # --- Failure Analysis ---
    print("\nFailure Analysis:")
    # Create binary error mask (1 for error, 0 for correct)
    errors = (val_preds != val_y).astype(int)

    # Calculate correlation between features and error
    # We use the processed val_df features
    feature_cols = [c for c in val_df.columns if c not in [ID_COL, TARGET_COL]]
    correlations = []

    for col in feature_cols:
        # Only check numeric columns (processed df should be all numeric)
        if pd.api.types.is_numeric_dtype(val_df[col]):
            col_values = val_df[col].values
            # Skip constant columns to avoid division by zero in correlation
            if np.std(col_values) == 0:
                continue

            # Compute Pearson correlation
            corr = np.corrcoef(col_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # --- Submission ---
    # Strict threshold check
    threshold = 0.9619111111111112
    if acc > threshold:
        print(f"\nValidation metric {acc} > {threshold}. Saving submission...")
        save_submission(test_preds, test_ids, SUBMISSION_PATH)
    else:
        print(f"\nValidation metric {acc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
