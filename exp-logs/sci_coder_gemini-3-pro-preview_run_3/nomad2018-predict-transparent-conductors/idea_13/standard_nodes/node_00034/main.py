import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Import from provided libraries
from library.config import setup_directories, TARGET_COLS
from library.data_handler import (
    load_metadata,
    build_feature_matrix,
    log_transform_targets,
)
from library.model_trainer import (
    run_training_pipeline,
    generate_submission_file,
    make_predictions,
)


def calculate_rmsle(y_true_log, y_pred_log):
    """
    Calculates RMSLE given log-transformed true and predicted values.
    Since y_log = log(1+y), RMSLE is simply RMSE of these log values.
    """
    mse = mean_squared_error(y_true_log, y_pred_log)
    return np.sqrt(mse)


def main():
    # 1. Setup
    setup_directories()

    # 2. Train Models
    # Using full dataset (sample_size=None) and caching
    print("Starting Training Pipeline...")
    models, feature_cols = run_training_pipeline(
        sample_size=None, load_cached_data=True
    )

    # 3. Validation and Failure Analysis
    print("\n--- Validation Assessment ---")
    val_meta = load_metadata("val")
    # Ensure features are built/loaded
    df_val = build_feature_matrix(val_meta, "val", load_cached_data=True)

    # Transform targets to log scale for metric calculation (consistent with training)
    df_val_trans = log_transform_targets(df_val, TARGET_COLS)

    # Select only the features used for training
    X_val = df_val_trans[feature_cols]

    rmsle_scores = []

    for target in TARGET_COLS:
        if target not in models:
            continue

        y_true_log = df_val_trans[target]
        y_pred_log = make_predictions(models[target], X_val)

        score = calculate_rmsle(y_true_log, y_pred_log)
        rmsle_scores.append(score)
        print(f"Target: {target}, RMSLE: {score:.6f}")

        # Failure Analysis: Correlation of error with features
        # Error magnitude
        error = np.abs(y_true_log - y_pred_log)

        # We calculate correlation of this error with all features
        # Add error to a temporary dataframe containing features
        analysis_df = X_val.copy()
        analysis_df["error_magnitude"] = error

        # Compute correlation matrix
        corr = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

        # Get top 5 positive correlations (features associated with high error)
        top_corr = corr.abs().sort_values(ascending=False).head(5)
        print(f"  Top 5 features correlated with error for {target}:")
        for feat, val in top_corr.items():
            print(f"    {feat}: {val:.4f}")

    # Final Metric: Column-wise RMSLE (Mean of RMSLEs)
    if rmsle_scores:
        final_metric = np.mean(rmsle_scores)
        print(f"Final Validation Metric: {final_metric}")

        # 4. Submission
        # Threshold check
        THRESHOLD = 0.057877

        if final_metric < THRESHOLD:
            print(
                f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission..."
            )
            generate_submission_file(
                models, feature_cols, sample_size=None, load_cached_data=True
            )
        else:
            print(
                f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Submission skipped."
            )
    else:
        print("No models trained. Skipping validation and submission.")


if __name__ == "__main__":
    main()
