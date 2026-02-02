import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.utils import WGS84Utils
from library.trainer import Trainer
from library.inference import InferencePipeline


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating mini metadata files.
    This prevents the pipeline from processing the entire dataset (GBs of data).
    """
    print("\n[1] Setting up Demo Environment...")

    # Override Config for Demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_SAVE_PATH = os.path.join(
        Config.WORKING_DIR, "submission", "submission.csv"
    )

    # Update Cache Paths to avoid conflicts
    Config.TRAIN_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "cache", "train_processed.parquet"
    )
    Config.VAL_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "cache", "val_processed.parquet"
    )
    Config.TEST_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "cache", "test_processed.parquet"
    )

    # Create directories
    os.makedirs(os.path.dirname(Config.TRAIN_CACHE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_SAVE_PATH), exist_ok=True)

    # Set Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process very few sequences
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Create Mini Metadata
    # We take the first N rows. Usually sorted by drive, so this gives one or partial trip.
    subset_size = 500

    # Train Metadata
    if os.path.exists(Config.TRAIN_METADATA_PATH):
        df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        mini_df = df.head(subset_size)
        mini_path = os.path.join(Config.WORKING_DIR, "mini_train_meta.csv")
        mini_df.to_csv(mini_path, index=False)
        Config.TRAIN_METADATA_PATH = mini_path
        print(f"  Created mini train metadata: {len(mini_df)} rows")

    # Val Metadata
    if os.path.exists(Config.VAL_METADATA_PATH):
        df = pd.read_csv(Config.VAL_METADATA_PATH)
        mini_df = df.head(min(len(df), 200))  # Smaller validation
        mini_path = os.path.join(Config.WORKING_DIR, "mini_val_meta.csv")
        mini_df.to_csv(mini_path, index=False)
        Config.VAL_METADATA_PATH = mini_path
        print(f"  Created mini val metadata: {len(mini_df)} rows")

    # Test Metadata
    if os.path.exists(Config.TEST_METADATA_PATH):
        df = pd.read_csv(Config.TEST_METADATA_PATH)
        mini_df = df.head(min(len(df), 200))
        mini_path = os.path.join(Config.WORKING_DIR, "mini_test_meta.csv")
        mini_df.to_csv(mini_path, index=False)
        Config.TEST_METADATA_PATH = mini_path
        print(f"  Created mini test metadata: {len(mini_df)} rows")


def verify_utils():
    """
    Verifies the WGS84 coordinate conversion utilities.
    """
    print("\n[2] Verifying Utilities...")
    wgs84 = WGS84Utils()

    # Test Case: 1 degree latitude change at equator is approx 110.574 km
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 1.0, 0.0

    d_north, d_east = wgs84.degrees_to_meters(lat2, lon2, lat1, lon1)

    print(f"  1 deg lat change -> North: {d_north:.2f}m, East: {d_east:.2f}m")

    # Assertion: Allow some variance due to ellipsoid calculation, but should be close to 110km
    assert 110000 < d_north < 112000, f"Latitude conversion error: {d_north}"
    assert abs(d_east) < 1e-5, f"Longitude conversion error (should be 0): {d_east}"

    # Inverse check
    d_lat, d_lon = wgs84.meters_to_degrees(d_north, d_east, lat1)
    print(f"  Inverse -> dLat: {d_lat:.6f}, dLon: {d_lon:.6f}")

    assert np.isclose(d_lat, 1.0), "Inverse latitude conversion failed"
    assert np.isclose(d_lon, 0.0), "Inverse longitude conversion failed"
    print("  Utils verification passed.")


def run_training_demo():
    """
    Instantiates the Trainer and runs a single epoch of training.
    """
    print("\n[3] Running Training Demo...")

    trainer = Trainer()

    # Verify model architecture instantiation
    print(f"  Model instantiated: {trainer.model.__class__.__name__}")

    # Run training loop
    # load_cached_data=False forces the preprocessor to read our new mini metadata
    # and generate a new parquet cache in the demo directory.
    trainer.fit(load_cached_data=False)

    # Verify output
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"  Training successful. Model saved at: {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model file was not created after training.")


def run_inference_demo():
    """
    Instantiates the InferencePipeline and generates a submission.
    """
    print("\n[4] Running Inference Demo...")

    pipeline = InferencePipeline()

    # Run inference
    pipeline.run(load_cached_data=False)

    # Verify output
    if os.path.exists(Config.SUBMISSION_SAVE_PATH):
        df = pd.read_csv(Config.SUBMISSION_SAVE_PATH)
        print(
            f"  Inference successful. Submission saved at: {Config.SUBMISSION_SAVE_PATH}"
        )
        print(f"  Submission shape: {df.shape}")

        # Basic checks
        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            col in df.columns for col in required_cols
        ), "Submission missing required columns"
        assert len(df) > 0, "Submission is empty"
    else:
        raise FileNotFoundError("Submission file was not created after inference.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        setup_demo_environment()
        verify_utils()
        run_training_demo()
        run_inference_demo()
        print("\n[SUCCESS] All demo components executed successfully.")
    except Exception as e:
        print(f"\n[FAILURE] Demo failed with error: {e}")
        raise e
