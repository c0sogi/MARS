import os
import sys
import numpy as np
import pandas as pd
import torch

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_auc
from library.feature_engineering import FeaturePipeline
from library.model_rf import train_rf_model, predict_rf_model
from library.model_mlp import train_mlp_model, predict_mlp_model


def run_failure_analysis(y_true, y_pred, val_df):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Select numerical columns for correlation analysis
    # We use the raw numerical columns from the validation dataframe
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target if present
    if Config.TARGET_COL in numeric_cols:
        numeric_cols.remove(Config.TARGET_COL)

    correlations = {}
    for col in numeric_cols:
        # Handle potential NaNs in raw data
        feat_values = val_df[col].fillna(0)
        # Ensure lengths match
        if len(feat_values) == len(errors):
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    return sorted_corr


def main():
    # 1. Initialization
    set_seed(Config.SEED)
    print("Starting execution...")

    # 2. Feature Engineering
    # We use the pipeline to generate/load features for both streams
    pipeline = FeaturePipeline()
    # load_cached_data=True allows using pre-computed features if available in ./working
    rf_features, mlp_features = pipeline.run(load_cached_data=True)

    # 3. Stream A: Random Forest
    print("\n--- Stream A: Random Forest ---")
    rf_model = train_rf_model(
        rf_features["X_train"],
        rf_features["y_train"],
        rf_features["X_val"],
        rf_features["y_val"],
    )

    # Generate RF predictions
    rf_val_preds = predict_rf_model(rf_model, rf_features["X_val"])
    rf_test_preds = predict_rf_model(rf_model, rf_features["X_test"])

    # 4. Stream B: Attention-Gated MLP
    print("\n--- Stream B: Attention-Gated MLP ---")
    # Train MLP
    mlp_model = train_mlp_model(mlp_features)

    # Generate MLP predictions
    # Note: predict_mlp_model handles device movement internally
    mlp_val_preds = predict_mlp_model(mlp_model, mlp_features, split="val")

    # For test set, we need to construct the features dictionary for the test split
    # The predict_mlp_model function expects keys like 'text_test', etc.
    # The pipeline output already has these keys.
    mlp_test_preds = predict_mlp_model(mlp_model, mlp_features, split="test")

    # 5. Ensemble
    print("\n--- Ensemble Aggregation ---")
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    # Weighted Average
    val_preds_ensemble = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)

    # 6. Validation Evaluation
    y_val_true = rf_features["y_val"]  # Same for both
    final_metric = compute_auc(y_val_true, val_preds_ensemble)

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Load validation dataframe to get raw feature values for correlation
    df_val = pd.read_csv(Config.VAL_PATH)
    run_failure_analysis(y_val_true, val_preds_ensemble, df_val)

    # 8. Submission Generation
    threshold = 0.6959737721862433
    if final_metric > threshold:
        print(
            f"\nValidation metric ({final_metric}) > threshold ({threshold}). Generating submission..."
        )

        # Ensemble Test Predictions
        test_preds_ensemble = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # Load Test IDs
        df_test = pd.read_csv(Config.TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test[Config.ID_COL],
                Config.TARGET_COL: test_preds_ensemble,
            }
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
