import sys
import os
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work correctly
sys.path.append(".")

# --- Patch tqdm to suppress progress bars as per requirements ---
# We monkey patch the tqdm object in the imported library modules
import library.odometry
import library.optimizer


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


library.odometry.tqdm = silent_tqdm
library.optimizer.tqdm = silent_tqdm

# Import library components
import library.config as config
from library.feature_engineering import extract_features
from library.odometry import extract_odometry
from library.model import ResidualRegressor
from library.optimizer import process_optimization


def main():
    print("=== Smartphone Locationing Pipeline Demonstration ===")

    # 1. Configuration for Speed
    print("\n[1] Configuring Hyperparameters for Speed...")
    # Reduce LightGBM estimators and complexity for quick training demonstration
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["min_child_samples"] = 5
    config.LGBM_PARAMS["num_leaves"] = 16
    config.LGBM_PARAMS["verbose"] = -1

    # Limit number of drives to process to ensure execution within time limit
    MAX_DRIVES_TRAIN = 2
    MAX_DRIVES_VAL = 1

    # 2. Feature Extraction (Training Data)
    print("\n[2] Extracting Features for Training Data...")
    # We use 'train' split. load_cached_data=False ensures we run the actual logic.
    df_train_feats = extract_features(
        split="train", load_cached_data=False, max_drives=MAX_DRIVES_TRAIN
    )

    print(f"    Train Features Shape: {df_train_feats.shape}")

    # Verification
    if df_train_feats.empty:
        raise RuntimeError(
            "Train features dataframe is empty. Check data availability."
        )

    required_cols = ["L1_E_force", "L5_N_vel_force", "WlsPositionXEcefMeters"]
    for col in required_cols:
        assert col in df_train_feats.columns, f"Missing feature column: {col}"
    assert (
        "LatitudeDegrees" in df_train_feats.columns
    ), "Ground truth missing in train features"

    # 3. Model Training
    print("\n[3] Training Residual Regressor...")
    model = ResidualRegressor()
    # Train with 2 folds to be quick
    model.train(df_train_feats, n_folds=2)

    assert len(model.models_east) == 2, "Failed to train East models"
    assert len(model.models_north) == 2, "Failed to train North models"

    # 4. Feature Extraction (Validation Data)
    print("\n[4] Extracting Features for Validation Data...")
    df_val_feats = extract_features(
        split="val", load_cached_data=False, max_drives=MAX_DRIVES_VAL
    )
    print(f"    Val Features Shape: {df_val_feats.shape}")

    if df_val_feats.empty:
        raise RuntimeError("Validation features dataframe is empty.")

    # 5. Prediction
    print("\n[5] Generating Predictions...")
    df_pred = model.predict(df_val_feats)

    print(f"    Predictions Shape: {df_pred.shape}")
    print(f"    Sample Prediction:\n{df_pred.head(2)}")

    assert "lat_pred" in df_pred.columns, "Prediction output missing lat_pred"
    assert "lon_pred" in df_pred.columns, "Prediction output missing lon_pred"

    # 6. Odometry Estimation
    print("\n[6] Estimating Odometry...")
    # Computes relative motion between epochs using TDCP and Doppler
    df_odom = extract_odometry(
        split="val", load_cached_data=False, max_drives=MAX_DRIVES_VAL
    )

    print(f"    Odometry Shape: {df_odom.shape}")
    print(f"    Sample Odometry:\n{df_odom.head(2)}")

    assert "odom_x" in df_odom.columns, "Odometry output missing odom_x"
    assert "reliability" in df_odom.columns, "Odometry output missing reliability"

    # 7. Graph Optimization
    print("\n[7] Running Graph Optimization...")
    # This step combines ML predictions (Anchors) and Odometry (Edges)
    # to produce a smooth, physically consistent trajectory.
    df_optimized = process_optimization(
        df_ml=df_pred,
        df_odom=df_odom,
        split="val",
        load_cached_data=False,
        max_drives=MAX_DRIVES_VAL,
    )

    print(f"    Optimized Trajectory Shape: {df_optimized.shape}")
    print(f"    Sample Optimized:\n{df_optimized.head(2)}")

    assert (
        "LatitudeDegrees" in df_optimized.columns
    ), "Optimized output missing LatitudeDegrees"
    assert (
        "LongitudeDegrees" in df_optimized.columns
    ), "Optimized output missing LongitudeDegrees"

    # Check if optimization actually changed values
    # Note: ML preds are in df_pred['lat_pred'], Optimized in df_optimized['LatitudeDegrees']
    comparison = pd.merge(df_pred, df_optimized, on=["tripId", "UnixTimeMillis"])
    diff = np.abs(comparison["lat_pred"] - comparison["LatitudeDegrees"]).sum()
    print(f"    Total Latitude Adjustment: {diff:.6f}")

    if diff == 0:
        print(
            "    Warning: Optimization did not adjust positions. This might happen with very short/clean segments."
        )

    print("\n=== Demonstration Successfully Completed ===")


if __name__ == "__main__":
    main()
