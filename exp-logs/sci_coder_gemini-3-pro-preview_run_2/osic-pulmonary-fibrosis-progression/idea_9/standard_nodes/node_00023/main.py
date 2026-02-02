import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_preparation import DataPreparation
from library.model_factory import QuantileGLMSystem


def main():
    # ==========================================
    # 1. Initialization & Setup
    # ==========================================
    print("Initializing Density-Aware Global-Local Quantile-GLM System...")

    # Set Random Seeds for Reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        print(f"CUDA is available. Using device: {Config.DEVICE}")
    else:
        print("CUDA not available. Using CPU.")

    # Setup directories
    Config.setup()

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    # The DataPreparation class handles feature extraction (cached) and matrix construction
    print("\n[Step 1] Preparing Data...")
    data_prep = DataPreparation()

    # Load Train and Validation Data
    # load_cached_data=True allows resuming from previous feature extraction steps
    X_fvc_train, X_unc_train, y_train, X_fvc_val, X_unc_val, y_val = (
        data_prep.get_train_val_data(load_cached_data=True)
    )

    print(f"  Training Samples: {len(y_train)}")
    print(f"  Validation Samples: {len(y_val)}")
    print(f"  FVC Feature Dim: {X_fvc_train.shape[1]}")
    print(f"  Uncertainty Feature Dim: {X_unc_train.shape[1]}")

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\n[Step 2] Training Models...")
    system = QuantileGLMSystem()

    # Fit the system (FVC Quantile Regressor + Uncertainty Gamma GLM)
    system.fit(X_fvc_train, X_unc_train, y_train)

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    print("\n[Step 3] Validating...")
    # Evaluate returns the modified Laplace Log Likelihood score
    val_score = system.evaluate(X_fvc_val, X_unc_val, y_val)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n[Step 4] Failure Analysis...")
    # Predict on validation set to analyze errors
    val_pred_fvc, _ = system.predict(X_fvc_val, X_unc_val)

    # Calculate Absolute Errors
    errors = np.abs(y_val - val_pred_fvc)

    # Calculate correlation between features and error magnitude
    # We iterate through features to find which ones correlate most with high error
    correlations = []
    n_features = X_fvc_val.shape[1]

    for i in range(n_features):
        feature_col = X_fvc_val[:, i]
        # Skip constant columns (std=0) to avoid division by zero in correlation
        if np.std(feature_col) > 1e-9:
            corr = np.corrcoef(feature_col, errors)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation (magnitude of impact)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("  Top 5 Features correlated with Error Magnitude (Systematic Bias Check):")
    for idx, corr in correlations[:5]:
        print(f"    Feature Index {idx}: Correlation = {corr:.6f}")

    print("  (Positive correlation implies higher feature values -> larger errors)")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    # Threshold defined in the task description logic
    THRESHOLD = -6.805292148096688

    if val_score > THRESHOLD:
        print(
            f"\n[Step 5] Validation Score ({val_score}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        X_fvc_test, X_unc_test, df_ids = data_prep.get_test_data(load_cached_data=True)

        # Generate Predictions
        pred_fvc, pred_delta = system.predict(X_fvc_test, X_unc_test)

        # Convert Delta (Expected MAE) to Confidence (Sigma)
        # For Laplace distribution: Sigma = MAE * sqrt(2)
        pred_sigma = pred_delta * np.sqrt(2)

        # Clip Confidence as per metric requirements
        pred_sigma = np.maximum(pred_sigma, 70)

        # Construct Submission DataFrame
        submission = df_ids.copy()
        submission["FVC"] = pred_fvc
        submission["Confidence"] = pred_sigma

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"  Submission saved to: {Config.SUBMISSION_PATH}")
        print("  Head of submission:")
        print(submission.head())

    else:
        print(
            f"\n[Step 5] Validation Score ({val_score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
