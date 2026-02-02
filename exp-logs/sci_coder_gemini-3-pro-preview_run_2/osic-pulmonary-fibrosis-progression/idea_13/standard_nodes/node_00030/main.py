import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.dataset_builder import DatasetBuilder
from library.regressors import QuantileMedianRegressor, ResidualUncertaintyRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing Pipeline...")
    Config.setup()
    seed_everything(Config.SEED)

    # 2. Data Loading & Feature Engineering
    # DatasetBuilder handles CNN extraction, Volumetrics, PCA, and Matrix construction
    print("Generating Datasets (this may take time if features are not cached)...")
    builder = DatasetBuilder()
    datasets = builder.generate_datasets(load_cached_data=True)

    train_data = datasets["train"]
    val_data = datasets["val"]
    test_data = datasets["test"]

    print(f"Train samples: {len(train_data['y'])}")
    print(f"Val samples: {len(val_data['y'])}")
    print(f"Test samples: {test_data['X_fvc'].shape[0]}")

    # 3. Training
    print("\n--- Training FVC Model (Quantile Regression) ---")
    # Train FVC Regressor (Median / q=0.5)
    fvc_model = QuantileMedianRegressor(
        alpha=0.01
    )  # Small regularization for stability
    fvc_model.fit(train_data["X_fvc"], train_data["y"])

    # Generate Training Residuals for Uncertainty Model
    train_preds = fvc_model.predict(train_data["X_fvc"])
    train_residuals = train_data["y"] - train_preds

    print("\n--- Training Uncertainty Model (ElasticNet on Residuals) ---")
    # Train Uncertainty Regressor to predict MAD (Mean Absolute Deviation)
    unc_model = ResidualUncertaintyRegressor(alpha=0.1, l1_ratio=0.5)
    unc_model.fit(train_data["X_unc"], train_residuals)

    # 4. Validation
    print("\n--- Validating ---")
    # Predict Median FVC
    val_fvc_pred = fvc_model.predict(val_data["X_fvc"])

    # Predict Uncertainty (MAD)
    val_mad_pred = unc_model.predict(val_data["X_unc"])

    # Convert MAD to Sigma for Laplace Metric
    # For Laplace distribution: Sigma = MAD * sqrt(2)
    val_sigma_pred = val_mad_pred * np.sqrt(2)

    # Calculate Metric
    metric_score = laplace_log_likelihood(val_data["y"], val_fvc_pred, val_sigma_pred)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {metric_score}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_meta = val_data["meta"].copy()
    val_meta["Predicted_FVC"] = val_fvc_pred
    val_meta["True_FVC"] = val_data["y"]
    val_meta["Abs_Error"] = np.abs(val_meta["True_FVC"] - val_meta["Predicted_FVC"])
    val_meta["Sigma"] = val_sigma_pred

    # Calculate correlations with error
    # We check numerical columns present in metadata
    analysis_cols = [
        "Age",
        "Weeks",
        "Percent",
        "Baseline_FVC",
        "Baseline_Percent",
        "Time_Delta",
    ]
    # Filter cols that actually exist in val_meta
    analysis_cols = [c for c in analysis_cols if c in val_meta.columns]

    print("Correlation between Absolute Error and Features:")
    correlations = (
        val_meta[analysis_cols + ["Abs_Error"]].corr()["Abs_Error"].drop("Abs_Error")
    )
    print(correlations.sort_values(ascending=False))

    # 6. Submission
    THRESHOLD = -6.805292148096688

    if metric_score > THRESHOLD:
        print(
            f"\nValidation metric ({metric_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_fvc_pred = fvc_model.predict(test_data["X_fvc"])
        test_mad_pred = unc_model.predict(test_data["X_unc"])
        test_sigma_pred = test_mad_pred * np.sqrt(2)

        # Prepare Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "Patient_Week": test_data["meta"]["Patient_Week"],
                "FVC": test_fvc_pred,
                "Confidence": test_sigma_pred,
            }
        )

        # Ensure correct formatting (FVC as int, Confidence as float/int)
        # Note: Sample submission usually expects FVC as int. Confidence can be float.
        # However, sample submission shows Confidence as int (100).
        # We will keep FVC as int. We will keep Confidence as float or round it?
        # The metric function handles floats. Let's keep high precision for Confidence
        # but FVC is physically discrete (ml), usually represented as int.
        sub_df["FVC"] = sub_df["FVC"].astype(int)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Preview
        print(sub_df.head())

    else:
        print(
            f"\nValidation metric ({metric_score}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
