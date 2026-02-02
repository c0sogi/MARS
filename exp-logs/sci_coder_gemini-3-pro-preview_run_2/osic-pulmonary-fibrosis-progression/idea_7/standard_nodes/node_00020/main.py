import os
import numpy as np
import pandas as pd
import warnings

# Import functions and constants from the provided library files
from library.config import SEED, VAL_META_PATH
from library.utils import seed_everything, laplace_log_likelihood
from library.pipeline import train_pipeline, inference_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Initialization
    seed_everything(SEED)
    print("Initializing Spatially-Aware Hybrid-Feature Quantile-GLM Pipeline...")

    # 2. Pipeline Execution
    # Orchestrates feature generation, preprocessing, and training.
    # load_cached_data=True allows using pre-computed features to speed up execution.
    fvc_model, unc_model, data_dict = train_pipeline(load_cached_data=True)

    # 3. Validation Assessment
    print("\n=== Validation Assessment ===")

    # Extract validation data from the dictionary returned by the pipeline
    X_fvc_val = data_dict["X_fvc_val"]
    y_val = data_dict["y_val"]
    X_unc_val = data_dict["X_unc_val"]

    # Generate predictions using the trained models
    # FVC Model predicts the median (q=0.5)
    val_preds_fvc = fvc_model.predict(X_fvc_val)

    # Uncertainty Model predicts the expected absolute error (Delta)
    val_preds_delta = unc_model.predict(X_unc_val)

    # Convert Delta to Sigma (Confidence) for the Laplace Metric
    # Analytical relationship: sigma = delta * sqrt(2)
    val_preds_sigma = val_preds_delta * np.sqrt(2)

    # Compute the evaluation metric
    metric = laplace_log_likelihood(y_val, val_preds_fvc, val_preds_sigma)

    # Print the metric in the strictly required format
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load validation metadata to correlate errors with interpretable clinical features
    if os.path.exists(VAL_META_PATH):
        df_val = pd.read_csv(VAL_META_PATH)

        # Ensure metadata aligns with the validation targets
        # The pipeline maintains order, but we check length for safety
        if len(df_val) == len(y_val):
            # Calculate Absolute Error
            abs_error = np.abs(y_val - val_preds_fvc)

            # Features to analyze for correlation with error
            features_to_analyze = ["Age", "Weeks", "Percent"]

            print("Correlation between Absolute Error and Clinical Features:")
            for feat in features_to_analyze:
                if feat in df_val.columns:
                    feat_values = df_val[feat].values

                    # Compute Pearson correlation using numpy
                    # We handle potential NaNs by creating a mask
                    valid_idx = ~np.isnan(feat_values) & ~np.isnan(abs_error)

                    if np.sum(valid_idx) > 1:
                        corr = np.corrcoef(
                            feat_values[valid_idx], abs_error[valid_idx]
                        )[0, 1]
                        print(f"  {feat}: {corr:.4f}")
        else:
            print(
                "Warning: Validation metadata length mismatch. Skipping detailed analysis."
            )
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # 5. Conditional Submission Generation
    # The threshold is specified in the task requirements
    SUBMISSION_THRESHOLD = -6.805292148096688

    if metric > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({metric}) meets threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission file...")
        inference_pipeline(fvc_model, unc_model, data_dict)
    else:
        print(
            f"\nValidation metric ({metric}) does not meet threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
