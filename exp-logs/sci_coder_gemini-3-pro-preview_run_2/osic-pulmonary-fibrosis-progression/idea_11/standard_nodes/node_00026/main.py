import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config, seed_everything
from library.feature_extraction import run_feature_extraction
from library.modeling import FVCRegressor, UncertaintyRegressor, calculate_metric

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Feature Extraction
    # This handles metadata loading, image processing (with caching),
    # tabular encoding, and PCA dimensionality reduction.
    print("Running feature extraction...")
    data_dict = run_feature_extraction(load_cached_data=True)

    # Unpack data
    train_df, X_train_pca = data_dict["train"]
    val_df, X_val_pca = data_dict["val"]
    test_df, X_test_pca = data_dict["test"]

    # 3. Preprocessing for Modeling
    # Extract targets and time variables
    y_train = train_df["FVC"].values.astype(np.float32)
    y_val = val_df["FVC"].values.astype(np.float32)

    # Weeks: Train/Val are already relative. Test needs adjustment.
    train_weeks = train_df["Weeks"].values.astype(np.float32)
    val_weeks = val_df["Weeks"].values.astype(np.float32)

    # Test weeks: Target Week - Baseline Week
    test_weeks_raw = test_df["Weeks"].values.astype(np.float32)
    test_baseline_weeks = test_df["Baseline_Weeks"].values.astype(np.float32)
    test_weeks = test_weeks_raw - test_baseline_weeks

    # 4. Train FVC Model (Median Regression)
    print("Training FVC Regressor...")
    fvc_model = FVCRegressor()
    fvc_model.fit(X_train_pca, train_weeks, y_train)

    # 5. Train Uncertainty Model (Residual Regression)
    print("Training Uncertainty Regressor...")
    # Generate in-sample predictions to get residuals
    train_preds = fvc_model.predict(X_train_pca, train_weeks)
    train_residuals = np.abs(y_train - train_preds)

    unc_model = UncertaintyRegressor()
    unc_model.fit(X_train_pca, train_weeks, train_residuals)

    # 6. Validation
    print("Running Validation...")
    val_preds_fvc = fvc_model.predict(X_val_pca, val_weeks)
    val_preds_mad = unc_model.predict(X_val_pca, val_weeks)

    # Convert MAD to Sigma (Analytical scaling for Laplace)
    # Sigma = MAD * sqrt(2)
    val_preds_sigma = val_preds_mad * np.sqrt(2)

    # Calculate Metric
    # Note: calculate_metric handles the clipping internally for scoring
    val_score = calculate_metric(y_val, val_preds_fvc, val_preds_sigma)

    # REQUIRED OUTPUT: Full precision validation metric
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    val_errors = np.abs(y_val - val_preds_fvc)

    # Create a temporary dataframe for correlation analysis
    analysis_df = val_df.copy()
    analysis_df["Error_Magnitude"] = val_errors

    # Select numerical features for correlation
    corr_features = ["Weeks", "Percent", "Age"]
    # Add 'Error_Magnitude' to the list
    corr_target = "Error_Magnitude"

    print(f"Correlation with {corr_target}:")
    for feat in corr_features:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df[corr_target])
            print(f"  {feat}: {corr:.4f}")

    # 8. Submission Generation
    # Threshold check
    THRESHOLD = -6.805292148096688

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_preds_fvc = fvc_model.predict(X_test_pca, test_weeks)
        test_preds_mad = unc_model.predict(X_test_pca, test_weeks)

        # Convert to Sigma
        test_preds_sigma = test_preds_mad * np.sqrt(2)

        # Apply Clipping for submission file
        # The prompt states: "confidence values are clipped at 70 ml"
        test_preds_sigma = np.maximum(test_preds_sigma, Config.SIGMA_MIN)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": test_preds_fvc,
                "Confidence": test_preds_sigma,
            }
        )

        # Format columns
        submission["FVC"] = submission["FVC"].astype(int)
        submission["Confidence"] = submission["Confidence"].round().astype(int)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print("Head of submission:")
        print(submission.head())

    else:
        print(
            f"\nValidation score ({val_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
