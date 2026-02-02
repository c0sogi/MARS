import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.workflow import prepare_dataset, train_stage_1, train_stage_2, inference

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Preparation
    # We use load_cached_data=True to utilize pre-computed features if available
    # The prepare_dataset function handles metadata loading, CNN feature extraction (or loading),
    # and feature merging.
    print("Loading and preparing datasets...")
    try:
        train_data = prepare_dataset("train", load_cached_data=True)
        val_data = prepare_dataset("val", load_cached_data=True)
        test_data = prepare_dataset("test", load_cached_data=True)
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)

    # 3. Model Training

    # Stage 1: FVC Prediction (Quantile Regression, q=0.5)
    # Uses Full features (Static + Time + Interactions)
    # Returns the trained model and predictions for train/val sets
    fvc_model, y_pred_train, y_pred_val = train_stage_1(
        train_data["X_full"], train_data["y"], val_data["X_full"], val_data["y"]
    )

    # Stage 2: Uncertainty Prediction (ElasticNet)
    # Uses Static features only to predict residuals
    # Returns the trained model
    unc_model = train_stage_2(
        train_data["X_static"],
        train_data["y"],
        y_pred_train,
        val_data["X_static"],
        val_data["y"],
        y_pred_val,
    )

    # 4. Validation Assessment
    # Recalculate metric explicitly to ensure we have the exact value for the condition
    mad_val = unc_model.predict(val_data["X_static"])
    sigma_val = mad_val * np.sqrt(2)
    final_metric = laplace_log_likelihood_metric(val_data["y"], y_pred_val, sigma_val)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors
    abs_errors = np.abs(val_data["y"] - y_pred_val)

    # Construct Feature Names
    # Static features come first, followed by PCA components
    feature_names = Config.STATIC_COLS + [
        f"PCA_{i}" for i in range(Config.PCA_COMPONENTS)
    ]

    # Ensure dimensions match before creating DataFrame
    X_val_static = val_data["X_static"]
    if X_val_static.shape[1] == len(feature_names):
        df_analysis = pd.DataFrame(X_val_static, columns=feature_names)
        df_analysis["Error_Magnitude"] = abs_errors

        # Calculate correlation
        correlations = df_analysis.corr()["Error_Magnitude"].drop("Error_Magnitude")

        # Sort by absolute correlation
        top_correlations = correlations.abs().sort_values(ascending=False).head(5)

        print("Top 5 Features correlated with Error Magnitude:")
        for feat, corr in top_correlations.items():
            # Get the original signed correlation
            signed_corr = correlations[feat]
            print(f"  {feat}: {signed_corr:.4f}")
    else:
        print("Skipping detailed feature correlation due to dimension mismatch.")
        print(f"Expected {len(feature_names)} features, got {X_val_static.shape[1]}.")

    # 6. Submission Generation
    # Threshold: -7.004077599888947
    THRESHOLD = -7.004077599888947

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        inference(
            fvc_model,
            unc_model,
            test_data["X_full"],
            test_data["X_static"],
            test_data["patient_weeks"],
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
