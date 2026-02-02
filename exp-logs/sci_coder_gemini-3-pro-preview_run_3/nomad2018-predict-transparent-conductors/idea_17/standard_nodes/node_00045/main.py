import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager

# Import from provided library files
from library.config import Config
from library.data_processor import DataPipeline
from library.model_trainer import DualTargetRegressor

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


@contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr (e.g., to hide tqdm progress bars from library calls).
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def set_seed(seed):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------


def main():
    # 1. Initialization
    set_seed(Config.RANDOM_SEED)
    print("Initializing Hybrid Geometric-Sublattice Fingerprinting Pipeline...")

    # 2. Data Loading & Feature Extraction
    # We suppress stderr to keep the output clean of progress bars from the library
    print("Loading data and generating features...")
    pipeline = DataPipeline()
    with suppress_stderr():
        train_df, val_df, test_df = pipeline.load_data(load_cached_data=True)

    print(
        f"Data loaded. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    # 3. Model Training
    # Initialize the dual target regressor (XGBoost)
    trainer = DualTargetRegressor()

    # Fit the models
    # Note: Training output is controlled by VERBOSE_EVAL in Config
    trainer.fit(train_df, val_df)

    # 4. Validation Assessment
    # Evaluate and compute the RMSLE metric
    metric = trainer.evaluate(val_df)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Generate predictions on the validation set
    val_preds_df = trainer.predict(val_df)

    # Merge predictions with ground truth using 'id'
    val_analysis = pd.merge(val_df, val_preds_df, on="id", suffixes=("_true", "_pred"))

    # Calculate error magnitudes
    # Metric is RMSLE, so we analyze the absolute error of the log-transformed targets
    targets = Config.TARGET_COLS
    error_cols = []

    for target in targets:
        # Transform to log space (log1p) as per the metric definition
        y_true_log = np.log1p(val_analysis[f"{target}_true"])
        y_pred_log = np.log1p(val_analysis[f"{target}_pred"])

        error_col = f"{target}_error"
        val_analysis[error_col] = np.abs(y_true_log - y_pred_log)
        error_cols.append(error_col)

    # Identify feature columns for correlation analysis
    # Exclude metadata, targets, and prediction columns
    exclude_cols = (
        ["id", "file_path"]
        + [f"{t}_true" for t in targets]
        + [f"{t}_pred" for t in targets]
        + targets
        + error_cols
    )

    feature_cols = [
        c
        for c in val_analysis.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(val_analysis[c])
    ]

    # Compute and print correlations
    if feature_cols:
        print(
            f"Analyzing correlations between errors and {len(feature_cols)} features..."
        )
        # Compute correlation matrix
        correlations = val_analysis[feature_cols + error_cols].corr()

        for err_col in error_cols:
            print(f"\nTop 10 features correlated with {err_col}:")
            # Extract correlations with the specific error column, sort by absolute value
            err_corr = (
                correlations[err_col]
                .drop(error_cols, errors="ignore")
                .abs()
                .sort_values(ascending=False)
            )
            print(err_corr.head(10))
    else:
        print("No numeric features available for correlation analysis.")

    # 6. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 0.056919346405286564

    if metric < THRESHOLD:
        print(f"\nValidation metric {metric} meets threshold {THRESHOLD}.")
        print("Generating final submission...")

        # Predict on test set
        submission_df = trainer.predict(test_df)

        # Save submission
        trainer.save_submission(submission_df)
    else:
        print(f"\nValidation metric {metric} is NOT lower than threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
