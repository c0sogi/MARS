import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import (
    set_seed,
    compute_score,
    save_submission,
    load_dataset,
    print_metric,
)
from library.feature_engineering import DataPreparer
from library.stacking_trainer import StackingEnsemble


def run_failure_analysis(y_true, y_pred, val_df):
    """
    Analyzes the correlation between prediction error and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error (absolute difference)
    errors = np.abs(y_true - y_pred)

    # Select numerical columns for correlation analysis
    # We exclude ID columns and the target itself
    numerical_cols = val_df.select_dtypes(include=["number"]).columns.tolist()
    exclude = [
        Config.TARGET_COL,
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numerical_cols = [c for c in numerical_cols if c not in exclude]

    correlations = []
    for col in numerical_cols:
        # Handle NaNs just in case, though metadata should be clean or handled
        feat_values = val_df[col].fillna(val_df[col].median())

        # Ensure lengths match
        if len(feat_values) != len(errors):
            continue

        # Compute Pearson correlation
        corr, _ = pearsonr(feat_values, errors)
        correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Modify Config for a faster baseline run if necessary
    # Given the small dataset size (approx 2.3k train), standard settings are acceptable.
    # However, we ensure DEBUG is False to run on full data.
    Config.DEBUG = False

    print("Initializing Data Preparer...")
    preparer = DataPreparer()

    # 2. Load and Prepare Data
    # We use load_cached_data=True to leverage any existing artifacts
    print("Loading Training Data...")
    X_lex_train, X_beh_train, X_sem_train, y_train, _ = preparer.get_features(
        "train", load_cached_data=True
    )

    print("Loading Validation Data...")
    X_lex_val, X_beh_val, X_sem_val, y_val, val_ids = preparer.get_features(
        "val", load_cached_data=True
    )

    # 3. Train Model
    print("Initializing Stacking Ensemble...")
    ensemble = StackingEnsemble()

    print("Fitting Model...")
    # The ensemble handles CV for the meta-learner internally
    ensemble.fit(X_lex_train, X_beh_train, X_sem_train, y_train)

    # 4. Validation
    print("Running Inference on Validation Set...")
    y_val_pred = ensemble.predict(X_lex_val, X_beh_val, X_sem_val)

    final_metric = compute_score(y_val, y_val_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Load raw validation dataframe to get interpretable features
    val_df = load_dataset("val")
    run_failure_analysis(y_val, y_val_pred, val_df)

    # 6. Submission
    THRESHOLD = 0.6913548345419015

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        print("Loading Test Data...")
        X_lex_test, X_beh_test, X_sem_test, _, test_ids = preparer.get_features(
            "test", load_cached_data=True
        )

        print("Predicting on Test Set...")
        y_test_pred = ensemble.predict(X_lex_test, X_beh_test, X_sem_test)

        save_submission(test_ids, y_test_pred)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
