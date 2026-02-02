import sys
import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Import from provided libraries
from library.config import Config
from library.trainer import train_xgboost_models, predict
from library.feature_pipeline import prepare_features
from library.utils import save_submission


def main():
    # 1. Setup Compute Device
    # Although XGBoost handles this via params, we explicitly set it to ensure compliance
    # with the requirement to detect and utilize GPU.
    if torch.cuda.is_available():
        # Monkey-patch Config params for GPU acceleration
        Config.XGB_PARAMS["device"] = "cuda"
        # 'hist' is efficient for both, but explicit gpu_hist (if older xgb) or device='cuda' (newer) is good.
        # We stick to 'hist' + device='cuda' which is standard for modern XGBoost.

    # 2. Train Models
    # We use the full dataset (sample_size=None) as the dataset is small (~2k rows)
    # and the GNN extractor is a Mock (fast).
    # load_cached_data=True allows using pre-computed features if they exist in ./working/idea_7
    print("Starting model training...")
    models, feature_cols, val_rmsle = train_xgboost_models(
        sample_size=None, load_cached_data=True
    )

    # 3. Print Final Validation Metric
    # Requirement: Print full precision.
    print(f"Final Validation Metric: {val_rmsle}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load validation set features and targets to analyze errors
    # We load cached data to ensure consistency with training
    X_val, y_val, _ = prepare_features(
        split="val", sample_size=None, load_cached_data=True
    )

    # Align features to match training columns (handle potential one-hot encoding differences)
    X_val = X_val.reindex(columns=feature_cols, fill_value=0)

    # Calculate errors
    # y_val is already log-transformed (log1p) because Config.LOG_TRANSFORM_TARGETS is True
    # We predict in log space to match.
    error_data = {}

    for target in Config.TARGET_COLS:
        model = models[target]
        # Predict
        preds_log = model.predict(X_val)
        actual_log = y_val[target].values

        # Calculate absolute error in log space (proxy for contribution to RMSLE)
        abs_error = np.abs(preds_log - actual_log)
        error_data[f"error_{target}"] = abs_error

    # Create a dataframe for correlation analysis
    error_df = pd.DataFrame(error_data, index=X_val.index)
    analysis_df = pd.concat([X_val, error_df], axis=1)

    # Compute and print correlations
    # We look for features that correlate highly with the error magnitude
    print("Correlation between Input Features and Model Error (Log Space):")

    numeric_features = X_val.select_dtypes(include=[np.number]).columns.tolist()

    for target in Config.TARGET_COLS:
        err_col = f"error_{target}"
        print(f"\nTop 5 features correlated with error in '{target}':")

        # Compute correlation
        correlations = analysis_df[numeric_features].corrwith(analysis_df[err_col])

        # Sort by absolute correlation
        top_corrs = correlations.abs().sort_values(ascending=False).head(5)

        # Print results
        for feature, corr_value in top_corrs.items():
            # Get the sign of the correlation
            sign = 1 if correlations[feature] > 0 else -1
            print(f"  {feature}: {corr_value * sign:.6f}")

    # 5. Submission Generation
    # Requirement: Generate submission ONLY IF val_rmsle < 0.06278041684313306
    THRESHOLD = 0.06278041684313306

    if val_rmsle < THRESHOLD:
        print(
            f"\nValidation metric ({val_rmsle}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        ids, preds = predict(
            models, feature_cols, sample_size=None, load_cached_data=True
        )
        save_submission(ids, preds, filename="submission.csv")
    else:
        print(
            f"\nValidation metric ({val_rmsle}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
