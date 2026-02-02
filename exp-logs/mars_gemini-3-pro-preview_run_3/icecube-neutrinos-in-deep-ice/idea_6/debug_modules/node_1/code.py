import os
import sys
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library import utils, data, model, train, inference


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("Starting Demonstration Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config for speed (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Small subset of events
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set up a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "model")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update file paths based on new dirs
    Config.MODEL_PATH = os.path.join(Config.MODEL_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup_directories()

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (Coordinate Transforms)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Spherical <-> Cartesian conversion
    # Case: Z-axis (Zenith=0)
    az_in, ze_in = 0.0, 0.0
    x, y, z = utils.spherical_to_cartesian(az_in, ze_in)
    print(f"    Spherical(0,0) -> Cartesian: ({x:.2f}, {y:.2f}, {z:.2f})")

    # Expect (0, 0, 1)
    assert np.isclose(x, 0.0, atol=1e-5), "X should be 0"
    assert np.isclose(y, 0.0, atol=1e-5), "Y should be 0"
    assert np.isclose(z, 1.0, atol=1e-5), "Z should be 1"

    # Convert back
    az_out, ze_out = utils.cartesian_to_spherical(x, y, z)
    print(f"    Cartesian(0,0,1) -> Spherical: az={az_out:.2f}, ze={ze_out:.2f}")

    assert np.isclose(ze_out, 0.0, atol=1e-5), "Zenith should be 0"
    # Azimuth is undefined/0 at pole, implementation returns 0 or similar

    # Case: X-axis (Azimuth=0, Zenith=pi/2)
    az_in, ze_in = 0.0, np.pi / 2
    x, y, z = utils.spherical_to_cartesian(az_in, ze_in)
    assert np.isclose(x, 1.0, atol=1e-5)
    assert np.isclose(z, 0.0, atol=1e-5)

    print("    Coordinate transformation logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Data Loaders...")

    # Get loaders
    train_loader, val_loader, test_loader = data.get_dataloaders()

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    print("    Batch Keys:", batch.keys())

    # Verify shapes
    # x: (B, N, 6)
    # pos: (B, N, 3)
    # mask: (B, N)
    # target: (B, 3)
    # rotation: (B, 3, 3)
    B = batch["x"].size(0)
    N = batch["x"].size(1)
    C = batch["x"].size(2)

    print(f"    Input Shape (x): {batch['x'].shape}")
    print(f"    Target Shape:    {batch['target'].shape}")

    assert B == Config.BATCH_SIZE or B <= Config.DEBUG_SIZE, "Batch size mismatch"
    assert C == 6, "Input channel dimension should be 6 (x, y, z, t, q, aux)"
    assert batch["target"].size(1) == 3, "Target should be a 3D vector"
    assert batch["rotation"].shape == (B, 3, 3), "Rotation matrix shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    cfdgn_model = model.CFDGN().to(device)

    # Move batch to device
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)

    # Forward pass
    print("    Performing forward pass...")
    pred = cfdgn_model(x, mask)

    print(f"    Prediction Shape: {pred.shape}")

    # Verify output
    assert pred.shape == (B, 3), "Output shape must be (Batch, 3)"

    # Verify normalization (Model output should be unit vectors)
    norms = torch.norm(pred, p=2, dim=1)
    print(f"    Mean Norm: {norms.mean().item():.4f}")
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4
    ), "Predictions are not normalized"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")

    optimizer = optim.AdamW(cfdgn_model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_mae = train.train_one_epoch(
            cfdgn_model, train_loader, optimizer, device
        )

        # Validate
        val_loss, val_mae = train.validate(cfdgn_model, val_loader, device)

        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

        # Basic assertion that loss is not NaN
        assert not np.isnan(train_loss), "Training loss is NaN"

    # Save the model
    torch.save(cfdgn_model.state_dict(), Config.MODEL_PATH)
    print(f"    Model saved to {Config.MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Use the inference module's logic
    # We pass the model directly since we already have it in memory
    df_submission = inference.predict_and_format(cfdgn_model, test_loader, device)

    print("    Inference complete.")
    print(f"    Submission Shape: {df_submission.shape}")
    print("    First 3 rows:")
    print(df_submission.head(3))

    # Verify submission format
    expected_cols = ["event_id", "azimuth", "zenith"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"

    # Verify value ranges
    assert df_submission["azimuth"].min() >= 0, "Azimuth cannot be negative"
    assert df_submission["azimuth"].max() <= 2 * np.pi, "Azimuth cannot exceed 2*pi"
    assert df_submission["zenith"].min() >= 0, "Zenith cannot be negative"
    assert df_submission["zenith"].max() <= np.pi, "Zenith cannot exceed pi"

    # Save to disk
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Check file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    print("\nDemonstration complete successfully.")


if __name__ == "__main__":
    main()
