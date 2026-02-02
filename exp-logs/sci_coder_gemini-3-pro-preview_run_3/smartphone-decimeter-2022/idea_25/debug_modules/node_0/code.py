import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Import library components
# Note: We assume the library files are in a package named 'library' in the current directory
import library.config as config
from library.gnss_utils import lla2ecef, ecef2lla, ecef2enu, enu2ecef
from library.data_loader import load_dataset
from library.model import LGBMEnsemble
from library.optimizer import TrajectoryOptimizer


def test_gnss_utils():
    """
    Demonstrates and verifies coordinate transformation functions.
    """
    print("\n[1/4] Testing GNSS Utils...")

    # Test Point: Googleplex (approximate)
    lat_ref, lon_ref, alt_ref = 37.422, -122.084, 10.0

    # 1. LLA -> ECEF
    x, y, z = lla2ecef(lat_ref, lon_ref, alt_ref)

    # 2. ECEF -> LLA (Round trip)
    lat_out, lon_out, alt_out = ecef2lla(x, y, z)

    # Validation
    assert np.isclose(lat_ref, lat_out, atol=1e-8), "Latitude round-trip failed"
    assert np.isclose(lon_ref, lon_out, atol=1e-8), "Longitude round-trip failed"
    assert np.isclose(alt_ref, alt_out, atol=1e-4), "Altitude round-trip failed"
    print("  - LLA <-> ECEF conversion verified.")

    # 3. ECEF -> ENU (Local tangent plane)
    # A point 100 meters East and 50 meters North of reference
    e_target, n_target, u_target = 100.0, 50.0, 5.0

    # Convert target ENU to ECEF
    x_t, y_t, z_t = enu2ecef(e_target, n_target, u_target, lat_ref, lon_ref, alt_ref)

    # Convert back to ENU
    e_out, n_out, u_out = ecef2enu(x_t, y_t, z_t, lat_ref, lon_ref, alt_ref)

    # Validation
    assert np.isclose(e_target, e_out, atol=1e-4), "Easting round-trip failed"
    assert np.isclose(n_target, n_out, atol=1e-4), "Northing round-trip failed"
    assert np.isclose(u_target, u_out, atol=1e-4), "Up round-trip failed"
    print("  - ECEF <-> ENU conversion verified.")


def test_data_loader():
    """
    Demonstrates loading a subset of the training data.
    """
    print("\n[2/4] Testing Data Loader...")

    # Load only 1 drive to ensure speed
    # load_cached_data=False forces processing from raw files
    df = load_dataset(split="train", max_drives=1, load_cached_data=False)

    if df.empty:
        raise RuntimeError(
            "Data loader returned empty DataFrame. Check input data availability."
        )

    print(f"  - Loaded {len(df)} rows.")
    print(f"  - Columns: {list(df.columns[:5])}...")

    # Validation
    required_cols = ["UnixTimeMillis", "Cn0DbHz_mean", "d_E", "d_N", "res_E", "res_N"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check for NaN in critical columns (some NaNs are expected in kinematics for first row, but features should be mostly present)
    assert df["Cn0DbHz_mean"].notna().sum() > 0, "Feature extraction failed (all NaNs)"

    print("  - Data structure verified.")
    return df


def test_model_training(train_df):
    """
    Demonstrates training the LightGBM ensemble.
    """
    print("\n[3/4] Testing Model Training...")

    # Modify config for speed
    config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees for demo
    config.N_FOLDS = 2  # Fewer folds

    model = LGBMEnsemble()

    # Train on the loaded subset
    # In a real scenario, we would split train/val properly, but here we demonstrate the API
    model.train(train_df)

    # Predict on the same set (just to verify prediction mechanics)
    preds = model.predict(train_df)

    print(f"  - Predictions shape: {preds.shape}")

    # Validation
    assert "pred_E" in preds.columns
    assert "pred_N" in preds.columns
    assert len(preds) == len(train_df)
    assert not preds.isna().all().all(), "Predictions resulted in all NaNs"

    print("  - Model training and prediction verified.")
    return preds


def test_optimizer(dataset_df, predictions_df):
    """
    Demonstrates the graph optimization phase.
    """
    print("\n[4/4] Testing Trajectory Optimizer...")

    optimizer = TrajectoryOptimizer()

    # The optimizer merges predictions with the dataset (which contains kinematics)
    # and solves the graph problem.
    optimized_df = optimizer.optimize(
        dataset_df, predictions_df, load_cached_data=False
    )

    print(f"  - Optimized result shape: {optimized_df.shape}")

    # Validation
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in optimized_df.columns, f"Optimizer output missing {col}"

    # Basic sanity check: Lat/Lon should be within valid range
    lat_min, lat_max = (
        optimized_df["LatitudeDegrees"].min(),
        optimized_df["LatitudeDegrees"].max(),
    )
    lon_min, lon_max = (
        optimized_df["LongitudeDegrees"].min(),
        optimized_df["LongitudeDegrees"].max(),
    )

    assert -90 <= lat_min <= lat_max <= 90, "Latitude out of bounds"
    assert -180 <= lon_min <= lon_max <= 180, "Longitude out of bounds"

    print(f"  - Output Lat range: [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"  - Output Lon range: [{lon_min:.4f}, {lon_max:.4f}]")
    print("  - Optimization verified.")

    return optimized_df


if __name__ == "__main__":
    print("Starting End-to-End Demonstration...")

    try:
        # 1. Verify Utils
        test_gnss_utils()

        # 2. Load Data
        df_train = test_data_loader()

        # 3. Train Model & Predict
        df_preds = test_model_training(df_train)

        # 4. Optimize Trajectory
        df_final = test_optimizer(df_train, df_preds)

        # Save dummy submission
        sub_path = os.path.join(config.OUTPUT_DIR, "demo_submission.csv")
        df_final.to_csv(sub_path, index=False)
        print(f"\nDemo completed successfully. Output saved to {sub_path}")

    except AssertionError as ae:
        print(f"\n[FAILED] Assertion Error: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
