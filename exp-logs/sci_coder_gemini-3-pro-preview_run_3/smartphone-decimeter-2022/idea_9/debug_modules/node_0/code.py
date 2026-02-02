import os
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import GeoUtils
from library.feature_engineering import FeatureGenerator
from library.model import LGBMResidualModel
from library.kalman_smoothing import KinematicKalmanSmoother


def run_demo():
    print("Starting Doppler-Aided Residual Boosting Pipeline Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Overriding Configuration for Speed...")
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["verbose"] = -1
    # Ensure working directory is clean-ish or just force overwrite logic
    # The library code creates directories if needed.
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Geospatial Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Geospatial Utilities...")
    lat, lon, alt = 37.421998, -122.084, 30.0
    x, y, z = GeoUtils.lla_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = GeoUtils.ecef_to_lla(x, y, z)

    print(f"Input LLA: {lat}, {lon}, {alt}")
    print(f"ECEF: {x:.2f}, {y:.2f}, {z:.2f}")
    print(f"Output LLA: {lat_out}, {lon_out}, {alt_out}")

    assert np.isclose(lat, lat_out, atol=1e-5), "Latitude round-trip failed"
    assert np.isclose(lon, lon_out, atol=1e-5), "Longitude round-trip failed"
    assert np.isclose(alt, alt_out, atol=1e-3), "Altitude round-trip failed"
    print("GeoUtils round-trip check passed.")

    # -------------------------------------------------------------------------
    # 3. Feature Generation (Train Split)
    # -------------------------------------------------------------------------
    print("\n[3] Generating Features for Training (Limit=1 drive)...")
    feat_gen = FeatureGenerator()

    # We use load_cached_data=False to demonstrate the pipeline logic
    # limit=1 ensures we only process one drive to keep it fast
    train_df = feat_gen.generate_features(
        split="train", load_cached_data=False, limit=1
    )

    print(f"Generated Train Features Shape: {train_df.shape}")

    # Assertions to check if features are actually generated
    expected_cols = [
        "v_east",
        "v_north",
        "speed",
        "signal_Cn0DbHz_mean",
        "target_east",
        "target_north",
    ]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing expected column: {col}"

    # Check for NaNs in critical columns (FeatureGenerator fills velocity NaNs with 0)
    assert not train_df["v_east"].isna().any(), "v_east contains NaNs"
    assert not train_df["target_east"].isna().any(), "target_east contains NaNs"
    print("Train feature generation verified.")

    # -------------------------------------------------------------------------
    # 4. Model Training
    # -------------------------------------------------------------------------
    print("\n[4] Training LightGBM Residual Models...")
    model = LGBMResidualModel()

    # For demo, use the same small dataset for train and val
    model.train(train_df, train_df)

    assert os.path.exists(model.model_east_path), "East model file not created"
    assert os.path.exists(model.model_north_path), "North model file not created"
    print("Model training and saving verified.")

    # -------------------------------------------------------------------------
    # 5. Feature Generation (Test Split)
    # -------------------------------------------------------------------------
    print("\n[5] Generating Features for Testing (Limit=1 drive)...")
    # Generate test features
    test_df = feat_gen.generate_features(split="test", load_cached_data=False, limit=1)

    print(f"Generated Test Features Shape: {test_df.shape}")

    # Test data shouldn't have targets
    assert "target_east" not in test_df.columns, "Test data should not have targets"
    assert "v_east" in test_df.columns, "Test data missing Doppler features"
    print("Test feature generation verified.")

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference...")
    preds_df = model.predict(test_df)

    print(f"Predictions Shape: {preds_df.shape}")
    assert "pred_east" in preds_df.columns, "Missing pred_east"
    assert "pred_north" in preds_df.columns, "Missing pred_north"
    print("Inference verified.")

    # -------------------------------------------------------------------------
    # 7. Kalman Smoothing
    # -------------------------------------------------------------------------
    print("\n[7] Applying Kinematic Kalman Smoothing...")

    # Prepare data for smoothing: Merge predictions with velocity features
    # We simulate the structure required by the smoother
    trip_ids = preds_df["tripId"].unique()
    if len(trip_ids) == 0:
        print("No trips found in predictions. Skipping smoothing check.")
    else:
        demo_trip_id = trip_ids[0]
        print(f"Smoothing trip: {demo_trip_id}")

        # Filter for one trip
        trip_preds = preds_df[preds_df["tripId"] == demo_trip_id]
        trip_feats = test_df[test_df["tripId"] == demo_trip_id]

        # Merge necessary columns
        trip_data = pd.merge(
            trip_preds,
            trip_feats[["tripId", "UnixTimeMillis", "v_east", "v_north"]],
            on=["tripId", "UnixTimeMillis"],
            how="left",
        )

        # Rename predictions to 'meas' as expected by the smoother logic in the library
        trip_data.rename(
            columns={"pred_east": "meas_east", "pred_north": "meas_north"}, inplace=True
        )

        smoother = KinematicKalmanSmoother()
        s_e, s_n = smoother.smooth_trip(trip_data)

        print(f"Smoothed Output Length: {len(s_e)}")

        assert len(s_e) == len(trip_data), "Smoothed output length mismatch"
        assert not np.isnan(s_e).any(), "Smoothed East contains NaNs"
        assert not np.isnan(s_n).any(), "Smoothed North contains NaNs"

        # Basic sanity check: Smoothing shouldn't diverge wildly from measurements in this stable demo
        # Mean difference
        diff_e = np.mean(np.abs(s_e - trip_data["meas_east"]))
        print(f"Mean smoothing adjustment (East): {diff_e:.4f} m")

        print("Kalman smoothing verified.")

    print("\n" + "=" * 60)
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
