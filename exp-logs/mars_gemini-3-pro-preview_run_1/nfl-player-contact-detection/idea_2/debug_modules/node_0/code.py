import os
import sys
import pandas as pd
import numpy as np
import gc
import warnings

# Import from the provided library
from library.config import Config
from library.feature_builder import FeatureBuilder
from library.model_factory import LGBMClassifierWrapper
from library.metrics import optimize_threshold, calculate_mcc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting NFL Contact Detection Library Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Initializing Configuration...")

    # Enable Debug mode to run on a small subset of data for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000  # Process 5000 rows for demonstration

    # Reduce model complexity for the demo to ensure quick execution
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["learning_rate"] = 0.1

    # Run setup to create directories and set seeds
    Config.setup()

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Path: {Config.MODEL_PATH}")

    # =========================================================================
    # 2. Feature Engineering
    # =========================================================================
    print("\n[2] Building Features...")

    builder = FeatureBuilder()

    # Build Training Features
    # We set load_cached_data=False to demonstrate the computation logic.
    # In a real run, this would default to True.
    print("-> Generating Training Features (Subset)...")
    df_train = builder.build_features(split="train", load_cached_data=False)

    # Validation: Check if features were generated
    assert not df_train.empty, "Training dataframe is empty!"
    assert "distance_lag0" in df_train.columns, "Feature 'distance_lag0' missing!"
    assert "contact" in df_train.columns, "Target 'contact' missing!"
    print(f"   Train Shape: {df_train.shape}")

    # Build Validation Features
    print("-> Generating Validation Features (Subset)...")
    df_val = builder.build_features(split="val", load_cached_data=False)

    assert not df_val.empty, "Validation dataframe is empty!"
    print(f"   Val Shape: {df_val.shape}")

    # =========================================================================
    # 3. Data Preparation
    # =========================================================================
    print("\n[3] Preparing Data for Training...")

    # Select feature columns (exclude metadata and target)
    # Features are those generated with lags or specific flags like 'is_ground'
    feature_cols = [c for c in df_train.columns if "lag" in c or c == "is_ground"]
    target_col = "contact"

    print(f"   Selected {len(feature_cols)} features.")

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # Clean up to save memory
    del df_train, df_val
    gc.collect()

    # =========================================================================
    # 4. Model Training
    # =========================================================================
    print("\n[4] Training LightGBM Model...")

    model_wrapper = LGBMClassifierWrapper()

    # Train the model
    # Note: We reduced n_estimators in Config earlier for speed
    model_wrapper.train(X_train, y_train, X_val, y_val)

    # Validation: Check if model file was created
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH}"
    print("   Model training complete and saved.")

    # =========================================================================
    # 5. Prediction & Evaluation
    # =========================================================================
    print("\n[5] Evaluating Model...")

    # Generate probability predictions on validation set
    y_pred_proba = model_wrapper.predict(X_val)

    # Validation: Check predictions shape and range
    assert len(y_pred_proba) == len(y_val), "Prediction length mismatch!"
    assert np.all(
        (y_pred_proba >= 0) & (y_pred_proba <= 1)
    ), "Probabilities out of range [0, 1]!"

    # Optimize Threshold
    print("   Optimizing threshold for MCC...")
    best_threshold, best_mcc = optimize_threshold(y_val, y_pred_proba, num_steps=50)

    print(f"   Best Threshold: {best_threshold:.4f}")
    print(f"   Best MCC Score: {best_mcc:.4f}")

    # Calculate MCC with a fixed threshold (e.g., 0.5) for comparison
    y_pred_binary_default = (y_pred_proba >= 0.5).astype(int)
    default_mcc = calculate_mcc(y_val, y_pred_binary_default)
    print(f"   Default (0.5) MCC: {default_mcc:.4f}")

    # Validation: Ensure MCC is a valid float
    assert -1.0 <= best_mcc <= 1.0, "MCC score out of valid range [-1, 1]"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
