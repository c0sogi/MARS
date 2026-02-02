import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import from the provided library files
from library.config import (
    SEED,
    DEVICE,
    WORKING_DIR,
    INPUT_CHANNELS,
    SEQ_LEN,
    seed_everything,
)
from library.utils import (
    angles_to_direction,
    direction_to_angles,
    angular_dist_score,
)
from library.model import TemporalCNN
from library.data_loader import get_dataloader
from library.trainer import train_model


def main():
    print("=== Starting IceCube Pipeline Demonstration ===\n")

    # 1. Setup and Reproducibility
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {SEED}")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utility Functions ---")

    # Test Case 1: Zenith=0 (North Pole/Up) -> x=0, y=0, z=1
    az_test, zen_test = 0.0, 0.0
    x, y, z = angles_to_direction(az_test, zen_test)
    print(f"Angles (0, 0) -> Vector ({x:.4f}, {y:.4f}, {z:.4f})")
    assert np.isclose(z, 1.0), "Zenith 0 should result in z=1"

    # Test Case 2: Zenith=pi/2, Azimuth=0 -> x=1, y=0, z=0
    az_test, zen_test = 0.0, np.pi / 2
    x, y, z = angles_to_direction(az_test, zen_test)
    print(f"Angles (0, pi/2) -> Vector ({x:.4f}, {y:.4f}, {z:.4f})")
    assert np.isclose(x, 1.0), "Zenith pi/2, Azimuth 0 should result in x=1"

    # Test Case 3: Round trip conversion
    az_orig, zen_orig = np.array([1.5]), np.array([2.0])
    x, y, z = angles_to_direction(az_orig, zen_orig)
    az_new, zen_new = direction_to_angles(x, y, z)
    assert np.allclose(az_orig, az_new) and np.allclose(
        zen_orig, zen_new
    ), "Round trip angle conversion failed"
    print("Utility functions verified successfully.")

    # 3. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    # Use a small sample size for speed
    train_loader = get_dataloader(
        metadata_path="./metadata/train_metadata.parquet",
        mode="train",
        max_samples=100,
        batch_size=16,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"Input Batch Shape: {inputs.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    # Assertions
    assert inputs.shape == (
        16,
        INPUT_CHANNELS,
        SEQ_LEN,
    ), f"Expected input shape (16, {INPUT_CHANNELS}, {SEQ_LEN}), got {inputs.shape}"
    assert targets.shape == (
        16,
        3,
    ), f"Expected target shape (16, 3), got {targets.shape}"
    print("Data loader verified successfully.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = TemporalCNN().to(DEVICE)

    # Move inputs to device
    inputs = inputs.to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        16,
        3,
    ), f"Expected output shape (16, 3), got {outputs.shape}"
    print("Model forward pass verified successfully.")

    # 5. Verify Training Pipeline
    print("\n--- Verifying Training Pipeline ---")
    # We use the provided trainer.train_model function
    # Limiting samples and epochs for rapid execution
    print("Starting short training run (1 epoch, 200 train samples, 50 val samples)...")

    best_model_path = train_model(max_train_samples=200, max_val_samples=50, epochs=1)

    print(f"Training completed. Best model saved to: {best_model_path}")
    assert os.path.exists(best_model_path), "Model checkpoint file was not created."

    # 6. Verify Inference Logic
    print("\n--- Verifying Inference Logic ---")
    # Note: We do not use library.inference.predict_and_submit() here because
    # it attempts to process the entire test set (13M events).
    # Instead, we simulate the inference steps on a small test subset.

    print("Simulating inference on small test subset (100 samples)...")
    test_loader = get_dataloader(
        metadata_path="./metadata/test_metadata.parquet",
        mode="test",
        max_samples=100,
        batch_size=16,
        num_workers=0,
    )

    # Load the trained model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    all_ids = []
    all_az = []
    all_zen = []

    with torch.no_grad():
        for inputs, event_ids in test_loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs)

            # Normalize
            preds_norm = F.normalize(preds, p=2, dim=1)

            # Convert to angles
            az, zen = direction_to_angles(
                preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
            )

            all_ids.extend(event_ids.numpy())
            all_az.extend(az.cpu().numpy())
            all_zen.extend(zen.cpu().numpy())

    # Create submission dataframe
    sub_df = pd.DataFrame({"event_id": all_ids, "azimuth": all_az, "zenith": all_zen})

    print("Inference simulation complete.")
    print(f"Generated predictions for {len(sub_df)} events.")
    print("Head of predictions:")
    print(sub_df.head())

    # Validation
    assert len(sub_df) == 100, "Did not generate predictions for all 100 test samples."
    assert not sub_df.isnull().values.any(), "Predictions contain NaN values."
    assert sub_df["azimuth"].between(0, 2 * np.pi).all(), "Azimuth values out of range."
    assert sub_df["zenith"].between(0, np.pi).all(), "Zenith values out of range."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
