import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided libraries
from library.feature_generator import generate_features
from library.model_trainer import GradientBoostingPredictor
from library.utils import calculate_rmsle, log_transform
from library.config import TARGET_COLS

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def perform_failure_analysis(val_df, preds_df, feature_cols):
    """
    Analyzes the correlation between prediction errors and input features.
    """
    print("\n" + "=" * 60)
    print(" FAILURE ANALYSIS")
    print("=" * 60)

    # Ensure indices align
    val_df = val_df.reset_index(drop=True)
    preds_df = preds_df.reset_index(drop=True)

    # Extract features
    X_val = val_df[feature_cols]

    # Drop non-numeric columns for correlation analysis
    X_val_numeric = X_val.select_dtypes(include=[np.number])

    for target in TARGET_COLS:
        print(f"\n--- Analysis for Target: {target} ---")

        y_true = val_df[target]
        y_pred = preds_df[target]

        # Calculate Log-Error Magnitude (aligned with RMSLE metric)
        # Error = |log(1+y_true) - log(1+y_pred)|
        log_true = np.log1p(y_true)
        log_pred = np.log1p(y_pred)
        error = np.abs(log_true - log_pred)

        print(f"Mean Absolute Log Error: {error.mean():.6f}")
        print(f"Max Absolute Log Error:  {error.max():.6f}")

        # Calculate correlation between error and features
        correlations = X_val_numeric.corrwith(error).sort_values(ascending=False)

        print("\nTop 5 Features correlated with higher error:")
        print(correlations.head(5))

        print("\nTop 5 Features correlated with lower error:")
        print(correlations.tail(5))


def main():
    print("Starting Hybrid Chemically-Resolved GNN-GBDT Pipeline...")

    # -------------------------------------------------------------------------
    # 1. Feature Generation / Loading
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading and generating features...")

    # Load Training Data
    # Cite debug_lesson_1: Invalidate Stale Cache to ensure full dataset usage
    train_df = generate_features("train", load_cached_data=False)
    print(f"Train features shape: {train_df.shape}")

    # Load Validation Data
    val_df = generate_features("val", load_cached_data=False)
    print(f"Validation features shape: {val_df.shape}")

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print("\n[2/5] Training models...")

    predictor = GradientBoostingPredictor()
    predictor.fit_model(train_df, val_df)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating on validation set...")

    # Generate predictions on validation set (returns original scale)
    val_preds_df = predictor.predict_values(val_df)

    # Calculate Final Validation Metric (RMSLE)
    # We calculate it manually here to ensure we print it exactly as requested
    y_true = val_df[TARGET_COLS].values
    y_pred = val_preds_df[TARGET_COLS].values

    final_metric = calculate_rmsle(y_true, y_pred)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[4/5] Performing failure analysis...")

    # Identify feature columns (exclude metadata)
    exclude_cols = ["id", "file_path"] + TARGET_COLS
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    perform_failure_analysis(val_df, val_preds_df, feature_cols)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5/5] Checking submission criteria...")

    THRESHOLD = 0.06278041684313306

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission."
        )

        # Load Test Data
        test_df = generate_features("test", load_cached_data=False)
        print(f"Test features shape: {test_df.shape}")

        # Generate Submission
        predictor.generate_submission(test_df)

    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
