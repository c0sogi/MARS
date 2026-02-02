import os
import sys
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error

# Import from provided libraries
from library.data_loader import load_metadata
from library.feature_engineering import FeaturePipeline
from library.model_handler import EnergyPredictor
from library.config import SUBMISSION_PATH, XGB_PARAMS


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error.
    Ensures non-negative values for log calculation.
    """
    # Clip predictions and true values to be non-negative
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def main():
    # Set seeds for reproducibility
    set_seed(42)

    print("Initializing Pipeline...")

    # 1. Initialize Feature Pipeline
    # This orchestrates the calculation of Electrostatic and Geometric features
    pipeline = FeaturePipeline()

    # 2. Load Metadata
    print("Loading Metadata...")
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")
    test_meta = load_metadata("test")

    # 3. Generate Features
    # We use load_cached_data=True to utilize precomputed features if available in ./working
    print("Generating/Loading Features...")
    # Note: FeaturePipeline.generate_features handles caching and merging with metadata
    train_df = pipeline.generate_features(train_meta, "train", load_cached_data=True)
    val_df = pipeline.generate_features(val_meta, "val", load_cached_data=True)
    test_df = pipeline.generate_features(test_meta, "test", load_cached_data=True)

    # 4. Configure Model
    # Check for GPU availability and update XGBoost parameters accordingly
    xgb_params = XGB_PARAMS.copy()
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost for GPU...")
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = "cuda"
    else:
        print("No GPU detected. Using CPU.")
        xgb_params["tree_method"] = "hist"
        # Ensure n_jobs is set for CPU parallelization
        xgb_params["n_jobs"] = -1

    # Initialize Predictor with configured parameters
    predictor = EnergyPredictor(xgb_params=xgb_params)

    # 5. Train Models
    # Train on the training set, using validation set for early stopping
    print("Training Models...")
    # The predictor handles log-transformation of targets internally
    predictor.train(train_df, val_df)

    # 6. Validation Assessment
    print("Running Validation...")
    # Predict returns values in original scale (inverse log applied)
    val_preds_df = predictor.predict(val_df)

    # Calculate Metrics
    targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    rmsle_scores = []

    for target in targets:
        y_true = val_df[target].values
        y_pred = val_preds_df[target].values

        score = calculate_rmsle(y_true, y_pred)
        rmsle_scores.append(score)
        print(f"Validation RMSLE for {target}: {score:.6f}")

    # Compute final metric as the average of the two RMSLE scores
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude per sample (mean of squared log errors)
    # Error_i = sqrt( 0.5 * sum_k (log(1+y_pred_ik) - log(1+y_true_ik))^2 )

    error_magnitudes = np.zeros(len(val_df))
    for target in targets:
        y_true = val_df[target].values
        y_pred = val_preds_df[target].values

        # Calculate squared log error for this target
        log_diff = np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_true, 0))
        error_magnitudes += log_diff**2

    # RMSLE per sample
    error_magnitudes = np.sqrt(error_magnitudes / len(targets))

    # Correlate error magnitude with input features
    # Exclude non-feature columns
    exclude_cols = ["id", "file_path"] + targets
    feature_cols = [c for c in val_df.columns if c not in exclude_cols]

    # Calculate correlations
    # Handle NaNs in features by filling with 0 for correlation calculation
    features_matrix = val_df[feature_cols].fillna(0)

    correlations = {}
    for col in feature_cols:
        # Only correlate with numeric features
        if pd.api.types.is_numeric_dtype(features_matrix[col]):
            # Check if column has variance
            if features_matrix[col].std() > 1e-9:
                corr = np.corrcoef(features_matrix[col], error_magnitudes)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for name, val in sorted_corrs[:5]:
        print(f"{name}: {val:.4f}")

    # 8. Submission
    # Generate submission only if validation metric meets the threshold
    THRESHOLD = 0.05095
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric:.6f} is below threshold {THRESHOLD}. Generating submission..."
        )
        predictor.save_submission(test_df, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {final_metric:.6f} is NOT below threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
