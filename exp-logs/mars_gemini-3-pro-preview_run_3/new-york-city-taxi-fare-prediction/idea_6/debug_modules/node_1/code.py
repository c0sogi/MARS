import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

# Import from the provided library
from library.config import Config
from library.utils import haversine_distance, rotate_coordinates
from library.data_processing import process_data, FeatureEngineer
from library.model_factory import ModelFactory
from library.training import run_training_pipeline
from library.inference import generate_ensemble_predictions


def main():
    # 1. Setup and Configuration Overrides for Speed
    print("=== Setting up demonstration environment ===")

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Override Config to run a fast, lightweight demonstration
    # We use a small sample size to ensure the script finishes quickly
    Config.DEBUG_SAMPLE_SIZE = 5000

    # Reduce model complexity for demonstration purposes
    Config.XGB_PARAMS.update(
        {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            # Keep device as cuda since A100 is available, but fallback handling isn't needed per prompt
        }
    )

    Config.LGBM_PARAMS.update(
        {"n_estimators": 10, "num_leaves": 31, "learning_rate": 0.1, "verbose": -1}
    )

    Config.EARLY_STOPPING_ROUNDS = 2

    # Clean working directory for a fresh run
    if os.path.exists(Config.WORKING_DIR):
        print(
            f"Note: Working directory {Config.WORKING_DIR} exists. Artifacts may be overwritten."
        )
    else:
        os.makedirs(Config.WORKING_DIR)

    # 2. Verify Utility Logic
    print("\n=== Verifying Utility Functions ===")

    # Test Haversine Distance
    # Distance between same point should be 0
    dist_zero = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
    assert np.isclose(
        dist_zero, 0.0
    ), f"Haversine distance for same point should be 0, got {dist_zero}"

    # Distance between NYC and London (approx 5570 km)
    # NYC: 40.7128 N, 74.0060 W
    # London: 51.5074 N, 0.1278 W
    dist_nyc_lon = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert (
        5500 < dist_nyc_lon < 5650
    ), f"Haversine distance calculation seems off: {dist_nyc_lon}"
    print("Haversine distance logic verified.")

    # Test Coordinate Rotation
    # Rotating (1, 0) by 90 degrees (pi/2) should be approx (0, 1)
    # Note: Function returns (y_new, x_new) -> (lat_new, lon_new)
    # Input is (lat=0, lon=1) -> (y=0, x=1)
    # Rotated: x' = 1*cos(90) - 0*sin(90) = 0
    #          y' = 1*sin(90) + 0*cos(90) = 1
    # Result should be (1, 0)
    lat_rot, lon_rot = rotate_coordinates(0, 1, np.pi / 2)
    assert np.isclose(lat_rot, 1.0) and np.isclose(
        lon_rot, 0.0
    ), f"Rotation logic failed. Expected (1.0, 0.0), got ({lat_rot}, {lon_rot})"
    print("Coordinate rotation logic verified.")

    # 3. Verify Feature Engineering Class
    print("\n=== Verifying Feature Engineering ===")

    # Create a dummy dataframe
    dummy_data = pd.DataFrame(
        {
            "pickup_datetime": ["2015-01-01 12:00:00", "2015-01-01 13:00:00"],
            "pickup_latitude": [40.7128, 40.7580],
            "pickup_longitude": [-74.0060, -73.9855],
            "dropoff_latitude": [40.7580, 40.7128],
            "dropoff_longitude": [-73.9855, -74.0060],
            "passenger_count": [1, 2],
        }
    )

    fe = FeatureEngineer()
    processed_dummy = fe.transform(dummy_data.copy())

    # Check if new columns exist
    expected_cols = [
        "hour_sin",
        "haversine_dist",
        "rotated_manhattan_dist",
        "dist_pickup_to_JFK",
    ]
    for col in expected_cols:
        assert (
            col in processed_dummy.columns
        ), f"Feature {col} missing after engineering."

    # Check if pickup_datetime was dropped (as per implementation)
    assert (
        "pickup_datetime" not in processed_dummy.columns
    ), "pickup_datetime should have been dropped."
    print("Feature Engineering logic verified.")

    # 4. Run Training Pipeline
    print("\n=== Running Training Pipeline ===")
    # This will process data, train models, and generate submission
    # We force load_cached_data=False to demonstrate the full flow
    run_training_pipeline(
        load_cached_data=False, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify artifacts
    xgb_model_path = os.path.join(Config.WORKING_DIR, "xgboost_model.joblib")
    lgbm_model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.joblib")

    assert os.path.exists(xgb_model_path), "XGBoost model file was not created."
    assert os.path.exists(lgbm_model_path), "LightGBM model file was not created."
    print("Training pipeline completed and models saved.")

    # 5. Run Inference Pipeline
    print("\n=== Running Inference Pipeline ===")
    # This simulates a separate inference run using the saved models
    generate_ensemble_predictions(
        load_cached_data=True, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify Submission
    submission_path = Config.SUBMISSION_FILE_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Basic validation of submission
    assert "key" in df_sub.columns, "Submission missing 'key' column."
    assert "fare_amount" in df_sub.columns, "Submission missing 'fare_amount' column."

    # Check for NaNs
    nan_count = df_sub["fare_amount"].isnull().sum()
    assert nan_count == 0, f"Submission contains {nan_count} NaNs in fare_amount."

    # Check for negative fares (simple sanity check, though models might output them if not constrained)
    # If models are reasonable, mean should be positive
    mean_fare = df_sub["fare_amount"].mean()
    print(f"Mean predicted fare: {mean_fare:.2f}")
    assert mean_fare > 0, "Mean predicted fare is non-positive, something is wrong."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
