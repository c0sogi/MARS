import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Import from provided library files
from library.model_pipeline import LungFunctionPredictor
from library.utils import laplace_log_likelihood, seed_everything
from library.config import SEED

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Initialization
    print("Initializing Multi-Axis Variance-Hybrid Quantile-Elastic Pipeline...")
    seed_everything(SEED)

    # Instantiate the pipeline
    predictor = LungFunctionPredictor()

    # 2. Data Preparation
    # load_cached_data=True ensures we use pre-computed features if they exist in ./working
    print("Preparing data...")
    data = predictor.prepare_data(load_cached_data=True)

    # 3. Model Training
    # Fits the Linear Quantile Regressor (FVC) and Elastic Net (Uncertainty)
    print("Training models...")
    predictor.fit(data)

    # 4. Validation Assessment
    print("\n=== Validation Assessment ===")

    # Retrieve validation data
    X_fvc_val = data["X_fvc_val"]
    X_unc_val = data["X_unc_val"]
    y_val = data["y_val"]

    # Generate predictions on validation set
    # Predict Median FVC
    y_pred_val = predictor.fvc_model.predict(X_fvc_val)

    # Predict Uncertainty (MAD)
    mad_pred_val = predictor.unc_model.predict(X_unc_val)

    # Convert MAD to Sigma (Confidence)
    # Analytical scaling: sigma = MAD * sqrt(2)
    sigma_pred_val = mad_pred_val * np.sqrt(2)

    # Calculate Metric
    val_score = laplace_log_likelihood(y_val, y_pred_val, sigma_pred_val)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors
    abs_errors = np.abs(y_val - y_pred_val)

    # Get the validation dataframe to correlate errors with clinical features
    df_val = data["df_val"]

    # Features to analyze
    analysis_cols = ["Weeks", "Age", "Percent", "FVC"]

    print("Correlation between Absolute Error and Features:")
    for col in analysis_cols:
        if col in df_val.columns:
            # Ensure alignment (df_val might have different indexing if not careful,
            # but prepare_data returns the aligned dataframe subset)
            feat_values = df_val[col].values

            # Calculate correlation
            if len(np.unique(feat_values)) > 1:
                corr, _ = pearsonr(abs_errors, feat_values)
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: N/A (Constant value)")

    # Check correlation with predicted uncertainty (Calibration check)
    # Ideally, higher predicted uncertainty should correlate with higher actual error
    corr_calib, _ = pearsonr(abs_errors, sigma_pred_val)
    print(f"  Predicted Confidence (Calibration): {corr_calib:.4f}")

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = -6.805292148096688

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predictor.predict(data)
    else:
        print(
            f"\nValidation score ({val_score}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
