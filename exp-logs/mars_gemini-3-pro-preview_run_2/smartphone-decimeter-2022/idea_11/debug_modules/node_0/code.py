import os
import pandas as pd
import torch
import shutil

# Import library components
from library.config import Config
from library.train import train_model
from library.inference import generate_submission


def main():
    print("Starting Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo
    # -------------------------------------------------------------------------
    print("Configuring demo parameters...")

    # Set a specific working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = demo_working_dir
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 64  # Smaller batch size
    Config.DEBUG_SAMPLE_SIZE = 0.02  # Use 2% of training data for speed

    # Update dependent paths in Config since they are static class attributes
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_data.parquet")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_data.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_data.parquet")
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.json")
    Config.MODEL_PATH = os.path.join(
        Config.WORKING_DIR, "lat_model.pth"
    )  # Renaming to verify custom path usage

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Path: {Config.MODEL_PATH}")
    print(f"Sample Fraction: {Config.DEBUG_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Train Model
    # -------------------------------------------------------------------------
    print("\n[Step 1] Running Training Pipeline...")
    try:
        # load_cached_data=False forces processing from raw CSVs
        train_model(
            debug_sample_fraction=Config.DEBUG_SAMPLE_SIZE, load_cached_data=False
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify Training Artifacts
    if not os.path.exists(Config.MODEL_PATH):
        raise AssertionError(f"Model file was not created at {Config.MODEL_PATH}")

    if not os.path.exists(Config.SCALER_PATH):
        raise AssertionError(f"Scaler file was not created at {Config.SCALER_PATH}")

    print("Training artifacts verified.")

    # -------------------------------------------------------------------------
    # 3. Inference
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running Inference Pipeline...")
    try:
        # load_cached_data=False forces processing of test data
        generate_submission(load_cached_data=False, batch_size=Config.BATCH_SIZE)
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {df_sub.columns.tolist()}"
        )

    # Check for NaNs
    if df_sub.isnull().any().any():
        raise AssertionError("Submission contains NaN values.")

    print("\n[Success] Demo completed successfully.")


if __name__ == "__main__":
    main()
