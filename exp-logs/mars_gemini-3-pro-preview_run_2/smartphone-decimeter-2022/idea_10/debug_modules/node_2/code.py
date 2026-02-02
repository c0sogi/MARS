import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Configuration Override
# We must import config first and modify it before importing other modules
# that import values from it.
import library.config as config

print("Configuring parameters for demonstration...")
# Limit data processing to a small number of samples for speed
config.DEBUG_SAMPLE_SIZE = 200
# Reduce training parameters
config.NUM_EPOCHS = 1
config.BATCH_SIZE = 16
# Ensure we use the working directory for all outputs
config.TRAIN_CACHE_PATH = os.path.join(config.WORKING_DIR, "demo_train_cache.parquet")
config.VAL_CACHE_PATH = os.path.join(config.WORKING_DIR, "demo_val_cache.parquet")
config.TEST_CACHE_PATH = os.path.join(config.WORKING_DIR, "demo_test_cache.parquet")
config.MODEL_CHECKPOINT_PATH = os.path.join(config.WORKING_DIR, "demo_model.pth")
config.SCALER_PATH = os.path.join(config.WORKING_DIR, "demo_scaler.json")
config.SUBMISSION_OUTPUT_PATH = os.path.join(config.WORKING_DIR, "demo_submission.csv")

# Now import the rest of the library
from library import utils
from library import data_loader
from library import model
from library import trainer
from library import inference

import importlib

importlib.reload(utils)
importlib.reload(data_loader)
importlib.reload(model)
importlib.reload(trainer)
importlib.reload(inference)


def run_demonstration():
    print("\n=== Starting GNSS Positioning Pipeline Demonstration ===\n")

    # ---------------------------------------------------------
    # 2. Utility Verification
    # ---------------------------------------------------------
    print("--- Verifying Utility Functions ---")
    # Test Haversine Distance
    # Distance between New York (40.7128, -74.0060) and London (51.5074, -0.1278)
    # Expected approx 5570 km
    lat1, lon1 = 40.7128, -74.0060
    lat2, lon2 = 51.5074, -0.1278
    dist = utils.haversine_distance(lat1, lon1, lat2, lon2)
    print(f"Haversine Distance (NY -> London): {dist/1000:.2f} km")

    # Basic assertion to ensure logic is sound (allow some margin for ellipsoid vs sphere diffs)
    assert 5500 < dist / 1000 < 5650, "Haversine distance calculation is off!"
    print("Utility verification passed.\n")

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("--- Loading and Processing Data ---")
    # We force re-computation to demonstrate the processing logic
    train_loader, val_loader = data_loader.get_train_val_loaders(load_cached_data=False)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Verify batch structure
    sample_traj, sample_sky, sample_target = next(iter(train_loader))
    print(
        f"Sample Trajectory Batch Shape: {sample_traj.shape}"
    )  # (B, Channels, Window)
    print(f"Sample Sky Batch Shape: {sample_sky.shape}")  # (B, Features)
    print(f"Sample Target Batch Shape: {sample_target.shape}")  # (B, 2)

    assert (
        sample_traj.shape[0] == config.BATCH_SIZE
        or sample_traj.shape[0] == config.DEBUG_SAMPLE_SIZE
    ), f"Batch size mismatch! Expected {config.BATCH_SIZE}, got {sample_traj.shape[0]}"
    assert sample_target.shape[1] == 2, "Target should have 2 dimensions (d_lat, d_lon)"
    print("Data loading verified.\n")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("--- Initializing Model ---")
    sky_motion_model = model.SkyMotionModel()

    # Move to device
    device = config.DEVICE
    sky_motion_model.to(device)
    print(f"Model moved to {device}")

    # Test forward pass
    sample_traj = sample_traj.to(device)
    sample_sky = sample_sky.to(device)
    with torch.no_grad():
        output = sky_motion_model(sample_traj, sample_sky)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (sample_traj.size(0), 2), "Model output shape mismatch!"
    print("Model initialization verified.\n")

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------
    print("--- Starting Training Loop ---")
    # Initialize Trainer
    model_trainer = trainer.Trainer(sky_motion_model)

    # Run fit (1 epoch as configured)
    model_trainer.fit(train_loader, val_loader)

    # Check if model checkpoint exists
    if os.path.exists(config.MODEL_CHECKPOINT_PATH):
        print(f"Model checkpoint saved at: {config.MODEL_CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved!")
    print("Training verified.\n")

    # ---------------------------------------------------------
    # 6. Inference
    # ---------------------------------------------------------
    print("--- Generating Submission ---")
    # We use the inference module's logic but call it explicitly to control flow
    # Note: inference.generate_submission re-initializes model and trainer internally if not passed,
    # but we can pass our trained trainer/model if we modified the function signature.
    # The provided library function signature is generate_submission(load_cached_data=True).
    # It reloads the model from the checkpoint we just saved.

    inference.generate_submission(load_cached_data=False)

    # Verify submission file
    if os.path.exists(config.SUBMISSION_OUTPUT_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_OUTPUT_PATH)
        print(f"Submission file created at: {config.SUBMISSION_OUTPUT_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        print("Head of submission:")
        print(df_sub.head())

        # Basic checks
        assert "tripId" in df_sub.columns
        assert "UnixTimeMillis" in df_sub.columns
        assert "LatitudeDegrees" in df_sub.columns
        assert "LongitudeDegrees" in df_sub.columns
        # Check that we have rows (should be equal to DEBUG_SAMPLE_SIZE or slightly less due to filtering)
        assert len(df_sub) > 0, "Submission file is empty!"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("Inference verified.\n")
    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
