import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# Import from the provided library
from library.config import Config, set_seed, setup_directories
from library.utils import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
)
from library.data_loader import IceCubeDataset, get_dataloaders
from library.network import ADGN_Model
from library.training import Trainer
from library.inference import generate_submission


def run_demo():
    print("=" * 50)
    print("IceCube Pipeline Demonstration Script")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo Run...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Enable Debug/Fast mode
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Process only ~200 events
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2  # Use fewer workers for demo to reduce overhead
    Config.PATIENCE = 1

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    setup_directories()
    set_seed(42)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Coordinate Conversion
    az_true = np.array([0, np.pi / 2, np.pi])
    zen_true = np.array([np.pi / 2, np.pi / 2, 0])  # x-axis, y-axis, z-axis

    x, y, z = spherical_to_cartesian(az_true, zen_true)

    # Expected: (1,0,0), (0,1,0), (0,0,1)
    assert np.allclose(x, [1, 0, 0], atol=1e-6), "Spherical to Cartesian X failed"
    assert np.allclose(y, [0, 1, 0], atol=1e-6), "Spherical to Cartesian Y failed"
    assert np.allclose(z, [0, 0, 1], atol=1e-6), "Spherical to Cartesian Z failed"

    az_rec, zen_rec = cartesian_to_spherical(x, y, z)
    assert np.allclose(az_rec, az_true, atol=1e-6), "Round trip Azimuth failed"
    assert np.allclose(zen_rec, zen_true, atol=1e-6), "Round trip Zenith failed"

    # Test Angular Distance Score
    # Distance between (1,0,0) and (1,0,0) should be 0
    y_true = np.stack([az_true, zen_true], axis=1)
    score_perfect = angular_dist_score(y_true, y_true)
    assert np.isclose(
        score_perfect, 0.0, atol=1e-6
    ), "Metric failed for perfect prediction"

    # Distance between (1,0,0) and (-1,0,0) (pi radians)
    y_opp = np.array([[np.pi, np.pi / 2]])  # -x axis
    y_ref = np.array([[0, np.pi / 2]])  # +x axis
    score_opp = angular_dist_score(y_ref, y_opp)
    assert np.isclose(
        score_opp, np.pi, atol=1e-6
    ), "Metric failed for opposite direction"

    print("Utility functions verified successfully.")

    # ---------------------------------------------------------
    # 3. Verify Data Loading
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loading Pipeline...")

    # Instantiate Dataset (Train Mode)
    # limit_batches=1 ensures we only process one parquet file
    train_ds = IceCubeDataset(mode="train", limit_batches=1)

    print(f"Dataset Size (Events): {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset should not be empty"

    # Fetch a single sample
    X, priors, y, event_id = train_ds[0]

    # Check Shapes
    # X: (NUM_PULSES, 7)
    assert X.shape == (
        Config.NUM_PULSES,
        7,
    ), f"Expected X shape ({Config.NUM_PULSES}, 7), got {X.shape}"
    # Priors: (19,)
    assert priors.shape == (19,), f"Expected priors shape (19,), got {priors.shape}"
    # Target: (2,) -> azimuth, zenith
    assert y.shape == (2,), f"Expected target shape (2,), got {y.shape}"
    assert isinstance(event_id, int), "Event ID should be an integer"

    print(
        f"Sample shapes verified: X={tuple(X.shape)}, priors={tuple(priors.shape)}, y={tuple(y.shape)}"
    )

    # Verify DataLoader
    train_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, limit_train_batches=1, limit_val_batches=1
    )

    # Fetch one batch
    batch_X, batch_priors, batch_y, _ = next(iter(train_loader))
    assert batch_X.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_X.shape[1] == Config.NUM_PULSES, "Pulse dimension mismatch"

    print("Data Loader verified successfully.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = ADGN_Model().to(device)

    # Move batch to device
    batch_X = batch_X.to(device)
    batch_priors = batch_priors.to(device)

    # Forward Pass
    with torch.no_grad():
        preds = model(batch_X, batch_priors)

    # Check Output Shape: (Batch_Size, 3) for Cartesian vectors
    assert preds.shape == (
        Config.BATCH_SIZE,
        3,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 3), got {preds.shape}"

    # Check Normalization (Output vectors should be unit length)
    norms = torch.norm(preds, p=2, dim=1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), "Model outputs are not normalized unit vectors"

    print("Model architecture verified successfully.")

    # ---------------------------------------------------------
    # 5. Verify Training Loop
    # ---------------------------------------------------------
    print("\n[5] Verifying Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Run training
    # This will use the overridden Config (1 epoch, debug size)
    trainer.fit()

    # Check if model checkpoint was created
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved after training"
    print(f"Model successfully saved to {Config.MODEL_CHECKPOINT_PATH}")

    # ---------------------------------------------------------
    # 6. Verify Inference & Submission
    # ---------------------------------------------------------
    print("\n[6] Verifying Inference and Submission Generation...")

    # Generate submission using the trained model
    # Limit test batches to 1 for speed
    generate_submission(limit_batches=1)

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Validate Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    expected_cols = ["event_id", "azimuth", "zenith"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"
    assert len(df_sub) > 0, "Submission dataframe is empty"

    # Check value ranges
    assert (
        df_sub["azimuth"].min() >= 0 and df_sub["azimuth"].max() <= 2 * np.pi
    ), "Azimuth values out of range"
    assert (
        df_sub["zenith"].min() >= 0 and df_sub["zenith"].max() <= np.pi
    ), "Zenith values out of range"

    print("Inference and submission verified successfully.")

    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
