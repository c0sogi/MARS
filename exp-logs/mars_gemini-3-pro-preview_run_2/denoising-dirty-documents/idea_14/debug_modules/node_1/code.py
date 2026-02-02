import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_rmse, tiled_inference, apply_tta
from library.model import TSPCResUNet
from library.dataset import DenoisingDataset
from library.train import train_one_epoch, validate, InferenceWrapper


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Setting up demonstration...")

    # Modify Config for a fast demonstration run
    Config.EPOCHS = 1
    Config.PATCHES_PER_IMAGE = 2  # Reduce patches to minimize dataset size
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for simple script

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Model Instantiation & Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")
    model = TSPCResUNet().to(device)

    # Create a dummy input tensor: (Batch, Channel, Height, Width)
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Perform a forward pass
    res1, res2 = model(dummy_input)

    # Verify output shapes
    # The model returns residuals for two stages, both should match input shape
    assert (
        res1.shape == dummy_input.shape
    ), f"Stage 1 output shape mismatch. Expected {dummy_input.shape}, got {res1.shape}"
    assert (
        res2.shape == dummy_input.shape
    ), f"Stage 2 output shape mismatch. Expected {dummy_input.shape}, got {res2.shape}"

    print("Model forward pass successful. Output shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Dataset Loading & Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Dataset Loading ---")

    # Initialize Training Dataset (Patch-based)
    # We use load_cached_data=True to leverage existing cache if available
    train_dataset = DenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        root_dir=Config.INPUT_DIR,
        augment=True,
        train_mode=True,
        load_cached_data=True,
    )

    # Check dataset length
    # Expected: num_images (92) * patches_per_image (2) = 184
    print(f"Train dataset size: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Train dataset is empty."

    # Fetch a single training sample
    noisy_patch, clean_patch = train_dataset[0]

    # Verify patch dimensions (C, H, W)
    expected_patch_shape = (1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    assert (
        noisy_patch.shape == expected_patch_shape
    ), f"Noisy patch shape mismatch. Got {noisy_patch.shape}"
    assert (
        clean_patch.shape == expected_patch_shape
    ), f"Clean patch shape mismatch. Got {clean_patch.shape}"

    print("Train dataset sample verified.")

    # Initialize Validation Dataset (Full Image)
    val_dataset = DenoisingDataset(
        metadata_path=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        augment=False,
        train_mode=False,
        load_cached_data=True,
    )

    # Fetch a single validation sample
    # Returns: noisy_tensor, clean_tensor, id
    val_noisy, val_clean, val_id = val_dataset[0]

    # Verify it is a full image (Rank 3: C, H, W)
    assert val_noisy.dim() == 3, "Validation image should be 3-dimensional (C, H, W)"
    assert val_clean.dim() == 3, "Validation label should be 3-dimensional (C, H, W)"

    print(
        f"Validation dataset sample verified (ID: {val_id}, Shape: {val_noisy.shape})."
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Training Loop ---")

    # Create a small subset of the training data for speed
    train_subset = Subset(train_dataset, indices=range(8))
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Setup Optimizer and Loss
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training
    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    print(f"Training step completed. Loss: {loss:.6f}")
    assert isinstance(loss, float), "Returned loss is not a float."
    assert loss > 0, "Loss should be positive."

    # -------------------------------------------------------------------------
    # 5. Inference & TTA Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Inference & TTA ---")

    # Use the validation image loaded earlier
    test_input = val_noisy.to(device)  # (1, H, W)

    # Wrap model for inference (returns only final stage residual)
    infer_model = InferenceWrapper(model)

    # Test Tiled Inference
    # This splits the image into patches, processes them, and stitches them back
    output_tiled = tiled_inference(
        infer_model,
        test_input,
        patch_size=128,
        overlap_ratio=0.5,
        batch_size=2,
        device=device,
    )

    assert (
        output_tiled.shape == test_input.shape
    ), f"Tiled inference output shape mismatch. Expected {test_input.shape}, got {output_tiled.shape}"

    print("Tiled inference executed successfully.")

    # Test Test-Time Augmentation (TTA)
    # This runs inference on flipped/rotated versions and averages the result
    output_tta = apply_tta(
        infer_model, test_input, patch_size=128, overlap_ratio=0.5, device=device
    )

    assert (
        output_tta.shape == test_input.shape
    ), f"TTA output shape mismatch. Expected {test_input.shape}, got {output_tta.shape}"

    print("Test-Time Augmentation executed successfully.")

    # -------------------------------------------------------------------------
    # 6. Validation Logic Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Validation Logic ---")

    # Create a small subset of the validation data
    val_subset = Subset(val_dataset, indices=range(2))

    # Run validation function
    # This calculates RMSE over the subset
    val_rmse = validate(model, val_subset, device)

    print(f"Validation run completed. RMSE: {val_rmse:.6f}")
    assert isinstance(val_rmse, (float, np.floating)), "Validation RMSE is not a float."
    assert val_rmse >= 0, "RMSE cannot be negative."

    print("\nAll demonstrations passed successfully!")


if __name__ == "__main__":
    run_demo()
