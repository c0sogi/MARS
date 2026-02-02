import os
import sys
import pandas as pd
import numpy as np
import warnings

# Import library modules
import library.config as config
from library.spatial_ops import (
    clamp_coordinates,
    haversine_distance,
    manhattan_distance,
    add_rotated_coordinates,
    get_spatial_grid_id,
)
from library.feature_builder import PipelineProcessor
from library.model_trainer import XGBTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_small_dataset_snapshot():
    """
    Creates a small subset of the training data to speed up the demonstration.
    Updates the config module to point to this new file.
    """
    print("Creating small dataset snapshot for rapid demonstration...")

    # Read only the first 10,000 rows of the metadata train file
    # The original file is ~44M rows, reading it all would be too slow for a demo
    original_train_path = os.path.join(config.METADATA_DIR, "train.parquet")

    # Using pyarrow to read a small slice efficiently
    try:
        df_small = pd.read_parquet(original_train_path).head(10000)
    except Exception as e:
        # Fallback if pyarrow slicing isn't supported directly (though it usually is)
        df_small = pd.read_parquet(original_train_path)
        df_small = df_small.head(10000)

    # Save to working directory
    temp_train_path = os.path.join(config.WORKING_DIR, "temp_train_small.parquet")
    df_small.to_parquet(temp_train_path)

    # Patch the config module
    config.TRAIN_DATA_PATH = temp_train_path
    config.TRAIN_SAMPLE_SIZE = 5000  # Use 5000 for the learner subsample

    # Patch XGBoost params for speed
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["early_stopping_rounds"] = 5

    print(f"Config patched. TRAIN_DATA_PATH: {config.TRAIN_DATA_PATH}")
    print(f"Config patched. TRAIN_SAMPLE_SIZE: {config.TRAIN_SAMPLE_SIZE}")


def verify_spatial_ops():
    """
    Demonstrates and verifies the spatial operations library.
    """
    print("\n=== Verifying Spatial Operations ===")

    # Create dummy data
    # Point A: NYC (approx 40.7128, -74.0060)
    # Point B: Out of bounds (45.0, -80.0) -> Should be clamped
    data = {
        "pickup_latitude": [40.7128, 45.0],
        "pickup_longitude": [-74.0060, -80.0],
        "dropoff_latitude": [40.7580, 40.0],  # 40.7580 is Times Square approx
        "dropoff_longitude": [-73.9855, -70.0],
    }
    df = pd.DataFrame(data)

    # 1. Test Clamping
    clamped_df = clamp_coordinates(df)

    # Check Row 1 (Valid NYC) - Should remain roughly same
    assert np.isclose(clamped_df.loc[0, "pickup_latitude"], 40.7128)

    # Check Row 2 (Out of bounds) - Should be clamped to NYC_BOUNDING_BOX
    # lat_max = 42.0, lon_min = -75.0, lon_max = -72.0
    assert clamped_df.loc[1, "pickup_latitude"] == config.NYC_BOUNDING_BOX["lat_max"]
    assert clamped_df.loc[1, "pickup_longitude"] == config.NYC_BOUNDING_BOX["lon_min"]
    assert clamped_df.loc[1, "dropoff_longitude"] == config.NYC_BOUNDING_BOX["lon_max"]

    print("✓ clamp_coordinates passed.")

    # 2. Test Distance
    # Distance between Point A and Times Square (Row 0)
    # Approx 5km
    dist = haversine_distance(
        df.loc[0, "pickup_latitude"],
        df.loc[0, "pickup_longitude"],
        df.loc[0, "dropoff_latitude"],
        df.loc[0, "dropoff_longitude"],
    )
    assert (
        4.0 < dist < 6.0
    ), f"Haversine distance {dist} seems incorrect for NYC intra-city trip"
    print(f"✓ haversine_distance passed (Calculated: {dist:.4f} km).")

    # 3. Test Grid ID
    grid_ids = get_spatial_grid_id(df, precision=2)
    expected_id = "40.71_-74.01_40.76_-73.99"
    assert grid_ids[0] == expected_id, f"Grid ID mismatch. Got {grid_ids[0]}"
    print("✓ get_spatial_grid_id passed.")


def run_pipeline():
    """
    Runs the feature engineering pipeline using the PipelineProcessor.
    """
    print("\n=== Running Data Pipeline ===")

    processor = PipelineProcessor()

    # Force processing from scratch (load_cached_data=False) to ensure we use our patched small dataset
    # Note: We need to clear any existing cache files in working dir if they exist to be safe,
    # but load_cached_data=False handles the logic to ignore them.
    train_df, val_df, test_df = processor.process_data(load_cached_data=False)

    # Validations
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Check for generated features
    expected_features = [
        "dist_haversine",
        "dist_manhattan",
        "pickup_rot_lat",
        "route_mean_fare",
        "temporal_fare_rate",
    ]

    for feat in expected_features:
        assert feat in train_df.columns, f"Missing feature {feat} in train_df"
        assert feat in test_df.columns, f"Missing feature {feat} in test_df"

    # Check that route statistics are not all NaN (implies successful join)
    # Since we used the train set to generate global stats, at least some overlap should exist
    assert (
        train_df["route_mean_fare"].notna().all()
    ), "NaNs found in route_mean_fare (Train)"

    print("✓ Pipeline processing completed successfully.")
    return train_df, val_df, test_df


def run_training(train_df, val_df, test_df):
    """
    Trains the XGBoost model and generates predictions.
    """
    print("\n=== Running Model Training ===")

    trainer = XGBTrainer()

    # Verify config patch worked
    assert trainer.num_boost_round == 10, "Trainer did not pick up patched n_estimators"

    # Train
    trainer.train(train_df, val_df)

    # Predict
    predictions = trainer.predict(test_df)

    # Validations
    assert len(predictions) == len(test_df), "Prediction length mismatch"
    assert (
        predictions >= 2.50
    ).all(), "Predictions contain values below minimum fare ($2.50)"

    print(
        f"✓ Training and prediction completed. Mean Prediction: ${predictions.mean():.2f}"
    )

    return trainer, predictions


def generate_submission_file(trainer, test_df, predictions):
    """
    Generates the submission CSV.
    """
    print("\n=== Generating Submission ===")

    trainer.generate_submission(test_df, predictions)

    # Verify file existence and format
    assert os.path.exists(config.SUBMISSION_OUTPUT_PATH), "Submission file not found"

    sub_df = pd.read_csv(config.SUBMISSION_OUTPUT_PATH)
    assert (
        "key" in sub_df.columns and "fare_amount" in sub_df.columns
    ), "Submission columns mismatch"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    print(f"✓ Submission verified at {config.SUBMISSION_OUTPUT_PATH}")
    print("First 5 rows:")
    print(sub_df.head())


if __name__ == "__main__":
    set_seed(42)

    # 1. Prepare environment for speed
    create_small_dataset_snapshot()

    # 2. Verify utility functions
    verify_spatial_ops()

    # 3. Execute Pipeline
    train_df, val_df, test_df = run_pipeline()

    # 4. Train Model
    trainer, predictions = run_training(train_df, val_df, test_df)

    # 5. Create Submission
    generate_submission_file(trainer, test_df, predictions)

    print("\nAll tasks completed successfully.")
