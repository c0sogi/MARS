import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.data_manager import DataProcessor
from library.modeling import FVCPredictor, UncertaintyPredictor
from library.utils import calculate_metric, format_submission


def main():
    # 1. Setup and Initialization
    Config.setup()
    print("Starting PCA-Enhanced Quantile-Elastic Pipeline...")

    # 2. Data Loading & Preprocessing
    # The DataProcessor handles caching, PCA, and feature engineering internally.
    processor = DataProcessor()
    data = processor.process_data(load_cached_data=Config.LOAD_CACHED_DATA)

    # Unpack processed data
    X_inter_train = data[
        "X_inter_train"
    ]  # Features for FVC model (Static + Time + Interactions)
    X_unc_train = data[
        "X_unc_train"
    ]  # Features for Uncertainty model (Static + Abs(Time))
    y_train = data["y_train"]

    X_inter_val = data["X_inter_val"]
    X_unc_val = data["X_unc_val"]
    y_val = data["y_val"]

    X_inter_test = data["X_inter_test"]
    X_unc_test = data["X_unc_test"]
    test_ids = data["test_ids"]

    print(
        f"Data loaded. Train shape: {X_inter_train.shape}, Val shape: {X_inter_val.shape}"
    )

    # 3. Stage 1: Train FVC Predictor (Quantile Regression)
    print("Stage 1: Training FVC Predictor (Median Regression)...")
    fvc_model = FVCPredictor()
    fvc_model.fit(X_inter_train, y_train)

    # 4. Stage 2: Compute Residuals
    print("Stage 2: Computing Training Residuals...")
    # Predict on training set to find errors
    y_train_pred = fvc_model.predict(X_inter_train)
    # Calculate absolute residuals (L1 error)
    train_residuals = np.abs(y_train - y_train_pred)

    # 5. Stage 3: Train Uncertainty Predictor (Elastic Net)
    print("Stage 3: Training Uncertainty Predictor (Elastic Net)...")
    uncertainty_model = UncertaintyPredictor()
    # We predict the magnitude of error using static features + absolute time
    uncertainty_model.fit(X_unc_train, train_residuals)

    # 6. Validation Inference
    print("Performing Validation...")
    # Predict FVC (Median)
    y_val_pred = fvc_model.predict(X_inter_val)

    # Predict Uncertainty (MAD)
    val_mad = uncertainty_model.predict(X_unc_val)

    # Convert MAD to Sigma (Standard Deviation for Laplace)
    # For Laplace distribution, sigma = MAD * sqrt(2)
    val_sigma = val_mad * np.sqrt(2)

    # 7. Metric Calculation
    val_metric = calculate_metric(y_val, y_val_pred, val_sigma)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load validation metadata to correlate errors with interpretable features
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    if Config.DEBUG:
        val_meta = val_meta.head(Config.DEBUG_SAMPLE_SIZE)

    # Calculate absolute errors
    val_errors = np.abs(y_val - y_val_pred)

    # Select numerical columns for correlation analysis
    analysis_cols = ["Weeks", "Age", "Percent", "FVC"]
    print("Correlation between Absolute Error and Features:")

    for col in analysis_cols:
        if col in val_meta.columns:
            feat_values = val_meta[col].values
            # Handle potential NaNs if any (though data analysis showed none)
            valid_mask = ~np.isnan(feat_values)
            if np.sum(valid_mask) > 1:
                corr, _ = pearsonr(feat_values[valid_mask], val_errors[valid_mask])
                print(f"  {col}: {corr:.4f}")

    # Also check correlation with predicted uncertainty
    corr_sigma, _ = pearsonr(val_sigma, val_errors)
    print(f"  Predicted Sigma: {corr_sigma:.4f} (Should be positive)")

    # 9. Submission Generation
    # Threshold check
    THRESHOLD = -6.881048560145116

    if val_metric > THRESHOLD:
        print(
            f"\nMetric ({val_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        y_test_pred = fvc_model.predict(X_inter_test)
        test_mad = uncertainty_model.predict(X_unc_test)
        test_sigma = test_mad * np.sqrt(2)

        # Load test metadata to get Patient_Week mapping
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Ensure alignment (data_manager extracts test_ids from the same file)
        # We re-verify length just in case
        if len(test_df) != len(y_test_pred):
            print("Warning: Test DataFrame length mismatch with predictions.")

        # Format Submission
        format_submission(test_df, y_test_pred, test_sigma, Config.SUBMISSION_PATH)

    else:
        print(
            f"\nMetric ({val_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
