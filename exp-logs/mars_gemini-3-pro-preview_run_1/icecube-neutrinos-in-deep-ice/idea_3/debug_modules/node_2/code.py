import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided libraries
from library.config import Config, set_seed
from library.utils import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
)
from library.data import get_dataloaders
from library.model import HybridRecurrentDenseNet
from library.train import run_training_and_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_utilities():
    print("\n=== Testing Utilities ===")

    # Test 1: Spherical to Cartesian (Zenith=0 should be Z-axis)
    az, zen = np.array([0.0]), np.array([0.0])
    x, y, z = spherical_to_cartesian(az, zen)
    print(f"Spherical(0,0) -> Cartesian({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
    assert (
        np.isclose(x[0], 0.0) and np.isclose(y[0], 0.0) and np.isclose(z[0], 1.0)
    ), "Conversion failed: Zenith 0 should map to (0,0,1)"

    # Test 2: Cartesian to Spherical (X-axis should be Az=0, Zen=pi/2)
    x_in, y_in, z_in = np.array([1.0]), np.array([0.0]), np.array([0.0])
    az_out, zen_out = cartesian_to_spherical(x_in, y_in, z_in)
    print(f"Cartesian(1,0,0) -> Spherical(Az={az_out[0]:.2f}, Zen={zen_out[0]:.2f})")
    assert np.isclose(az_out[0], 0.0) and np.isclose(
        zen_out[0], np.pi / 2
    ), "Conversion failed: (1,0,0) should map to Az=0, Zen=pi/2"

    # Test 3: Angular Distance Score
    # Identical vectors -> 0 error
    y_true = np.array([[1.0, 1.0]])  # Random angles
    y_pred = np.array([[1.0, 1.0]])
    score = angular_dist_score(y_true, y_pred)
    print(f"Angular Error (Identical): {score:.6f}")
    assert np.isclose(score, 0.0), "Angular error for identical vectors should be 0"

    print("Utilities verification passed.")


def test_data_pipeline():
    print("\n=== Testing Data Pipeline ===")

    # Loaders are already configured via Config global state
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch one batch
    seq, features, targets = next(iter(train_loader))

    print(f"Sequence Shape: {seq.shape}")
    print(f"Features Shape: {features.shape}")
    print(f"Targets Shape: {targets.shape}")

    # Assertions
    # Seq: (Batch, Seq_Len, N_Features) -> (Batch, 196, 5)
    assert seq.shape[1] == 196, f"Expected Seq Len 196, got {seq.shape[1]}"
    assert seq.shape[2] == 5, f"Expected 5 Seq Features, got {seq.shape[2]}"

    # Manual Features: (Batch, N_Manual) -> (Batch, 6)
    assert (
        features.shape[1] == 6
    ), f"Expected 6 Manual Features, got {features.shape[1]}"

    # Targets: (Batch, 2) -> Azimuth, Zenith
    assert targets.shape[1] == 2, f"Expected 2 Targets, got {targets.shape[1]}"

    print("Data pipeline verification passed.")
    return seq, features


def test_model(seq, features):
    print("\n=== Testing Model Architecture ===")

    device = torch.device(Config.DEVICE)
    model = HybridRecurrentDenseNet().to(device)

    # Move inputs to device
    seq = seq.to(device)
    features = features.to(device)

    # Forward pass
    output = model(seq, features)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Output should be (Batch, 3) -> x, y, z vector
    assert output.shape == (
        seq.shape[0],
        3,
    ), f"Expected output shape ({seq.shape[0]}, 3), got {output.shape}"

    print("Model verification passed.")


def test_full_training_loop():
    print("\n=== Testing Full Training & Inference Loop ===")

    # Run the provided training script function
    # This handles training, validation, saving, and inference
    run_training_and_inference(load_cached_data=True)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check columns
    expected_cols = ["event_id", "azimuth", "zenith"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check values are within valid ranges
    assert (
        df_sub["azimuth"].min() >= 0 and df_sub["azimuth"].max() <= 2 * np.pi
    ), "Azimuth values out of range [0, 2pi]"
    assert (
        df_sub["zenith"].min() >= 0 and df_sub["zenith"].max() <= np.pi
    ), "Zenith values out of range [0, pi]"

    print("Full training loop verification passed.")


def main():
    # 1. Setup Environment & Config
    # We modify Config attributes to ensure the demo runs quickly and is isolated
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary subdirs
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed optimizations
    Config.MAX_SAMPLES = 200  # Tiny subset
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small test
    Config.PATIENCE = 1

    # Set seed
    set_seed(Config.SEED)

    print(f"Configured for demo run in: {Config.WORKING_DIR}")
    print(f"Using Device: {Config.DEVICE}")

    # 2. Run Tests
    test_utilities()

    # Get a batch for model testing
    seq_batch, feat_batch = test_data_pipeline()

    test_model(seq_batch, feat_batch)

    test_full_training_loop()

    # 3. Cleanup (Optional, but good practice for demos)
    print("\nCleaning up temporary files...")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    print("Cleanup complete.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
