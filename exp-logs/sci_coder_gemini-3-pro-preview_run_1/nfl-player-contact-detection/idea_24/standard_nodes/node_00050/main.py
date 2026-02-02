import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.workflow_manager import WorkflowManager
from library.data_loader import DataLoader
from library.model_factory import LGBMExpert, XGBExpert
from library.utils import load_cache, compute_mcc, seed_everything


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("Initializing Runfile...")

    # Modify Config for Fast Baseline Execution
    # Reducing boost rounds ensures training fits within the time limit
    Config.NUM_BOOST_ROUND = 500
    Config.VERBOSE_EVAL = -1

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Training Phase
    # ---------------------------------------------------------
    # Instantiate the workflow manager
    wm = WorkflowManager()

    # Run the training pipeline
    # debug=False ensures we use the full dataset for mining and validation
    # load_cached_data=True attempts to use pre-computed features if available
    try:
        wm.run_training_phase(debug=False, load_cached_data=True)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Starting Validation Assessment ---")

    # Load Validation Data
    loader = DataLoader()
    df_val = loader.get_val_data(debug=False, load_cached_data=True)

    if df_val is None or df_val.empty:
        print("Error: Validation data is empty.")
        sys.exit(1)

    X_val = df_val[Config.FEATURES]
    y_val = df_val["contact"].values

    # Load Trained Models
    print("Loading trained models for evaluation...")
    if not os.path.exists(wm.expert_lgbm_path) or not os.path.exists(
        wm.expert_xgb_path
    ):
        print("Error: Trained models not found.")
        sys.exit(1)

    lgbm_model = LGBMExpert.load(wm.expert_lgbm_path)
    xgb_model = XGBExpert.load(wm.expert_xgb_path)

    # Load Optimized Threshold
    if os.path.exists(wm.threshold_path):
        best_threshold = load_cache(wm.threshold_path)[0]
    else:
        best_threshold = 0.5
        print("Warning: Threshold file not found, using default 0.5")

    # Generate Predictions (Ensemble)
    # Note: Models handle GPU/CPU internal logic
    p_lgbm = lgbm_model.predict(X_val)
    p_xgb = xgb_model.predict(X_val)
    p_ensemble = (p_lgbm + p_xgb) / 2.0

    # Apply Threshold
    y_pred = (p_ensemble >= best_threshold).astype(int)

    # Compute Metric
    val_mcc = compute_mcc(y_val, y_pred)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_mcc}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate residuals (absolute error of probabilities)
    # Using continuous probability error gives better correlation signal than binary error
    residuals = np.abs(y_val - p_ensemble)

    # Compute correlation between features and residuals
    # High correlation implies the feature is associated with higher error rates
    print("Correlation between Feature Values and Prediction Error:")
    feature_corrs = {}
    for feature in Config.FEATURES:
        if feature in X_val.columns:
            # Handle potential NaNs just in case, though pipeline handles them
            feat_vals = X_val[feature].fillna(0)
            corr = np.corrcoef(feat_vals, residuals)[0, 1]
            feature_corrs[feature] = corr

    # Sort and print top correlations
    sorted_corrs = sorted(feature_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:10]:
        print(f"  {feat}: {corr:.4f}")

    # ---------------------------------------------------------
    # 4. Submission Generation
    # ---------------------------------------------------------
    TARGET_METRIC = 0.6865

    if val_mcc > TARGET_METRIC:
        print(
            f"\nValidation Metric ({val_mcc}) > Target ({TARGET_METRIC}). Generating Submission..."
        )
        try:
            wm.run_inference_phase(debug=False, load_cached_data=True)
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            sys.exit(1)
    else:
        print(
            f"\nValidation Metric ({val_mcc}) <= Target ({TARGET_METRIC}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
