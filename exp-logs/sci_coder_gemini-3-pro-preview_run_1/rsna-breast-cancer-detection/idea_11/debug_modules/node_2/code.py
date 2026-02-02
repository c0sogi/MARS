import sys
import gc
import torch


def cleanup_memory():
    sys.last_traceback = None
    sys.last_type = None
    sys.last_value = None
    # Cite debug_lesson_22: Exclude essential modules from cleanup to prevent NameError
    keep_list = ["cleanup_memory", "gc", "sys", "torch"]
    for name in list(globals().keys()):
        if not name.startswith("__") and name not in keep_list:
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


cleanup_memory()

import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from unittest.mock import patch

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import SiameseMammographyDataset, get_dataloaders
from library.model import PyramidSiameseEfficientNet
from library.train import train_one_epoch, validate


def run_demonstration():
    print("=" * 50)
    print("Starting Library Usage Demonstration")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment...")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config for Speed/Demo purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small batch for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.DEVICE = "cpu"  # Use CPU for simple demo to avoid GPU init overhead if busy
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"

    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utility Functions...")

    # Test Probabilistic F1
    y_true = np.array([1, 0, 1, 0])
    y_pred_good = np.array([0.9, 0.1, 0.8, 0.2])  # Good predictions
    y_pred_bad = np.array([0.1, 0.9, 0.2, 0.8])  # Bad predictions

    pf1_good = probabilistic_f1(y_true, y_pred_good)
    pf1_bad = probabilistic_f1(y_true, y_pred_bad)

    print(f"pF1 (Good Predictions): {pf1_good:.4f}")
    print(f"pF1 (Bad Predictions):  {pf1_bad:.4f}")

    assert pf1_good > 0.8, "pF1 calculation for good predictions seems incorrect."
    assert pf1_bad < 0.2, "pF1 calculation for bad predictions seems incorrect."
    print("Utility verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 3] Demonstrating Data Pipeline...")

    # Monkey-patch _load_image to avoid dependency on actual DICOM files/cv2 support
    # This ensures the demo runs purely on logic without I/O errors.
    def mock_load_image(self, rel_path):
        # Return a random noise image simulating a normalized float32 image
        # Size chosen to be larger than Config.IMG_SIZE to test resizing logic
        return np.random.rand(800, 800).astype(np.float32)

    with patch.object(
        SiameseMammographyDataset,
        "_load_image",
        side_effect=mock_load_image,
        autospec=True,
    ):

        # Initialize DataLoaders
        print("Initializing DataLoaders (with mocked image loading)...")
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

        # Fetch one batch
        inputs, targets = next(iter(train_loader))
        img_target, img_contra = inputs

        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Target Image Tensor Shape: {img_target.shape}")
        print(f"Contra Image Tensor Shape: {img_contra.shape}")
        print(f"Labels Shape: {targets.shape}")

        # Assertions
        # Expected shape: (B, 3, H, W) -> 3 channels are [Image, Age, Implant]
        expected_shape = (
            Config.BATCH_SIZE,
            Config.IN_CHANNELS,
            Config.IMG_SIZE[0],
            Config.IMG_SIZE[1],
        )
        assert (
            img_target.shape == expected_shape
        ), f"Expected {expected_shape}, got {img_target.shape}"
        assert (
            img_contra.shape == expected_shape
        ), f"Expected {expected_shape}, got {img_contra.shape}"
        assert targets.shape == (
            Config.BATCH_SIZE,
        ), f"Expected ({Config.BATCH_SIZE},), got {targets.shape}"

        # Verify Age/Implant Map consistency (Channels 1 and 2)
        # In a batch, for a single image, the age map (channel 1) should be constant spatially
        sample_idx = 0
        age_map = img_target[sample_idx, 1, :, :]
        implant_map = img_target[sample_idx, 2, :, :]

        assert torch.std(age_map) < 1e-6, "Age map should be spatially constant."
        assert (
            torch.std(implant_map) < 1e-6
        ), "Implant map should be spatially constant."

        print("Data Pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Demonstrating Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = PyramidSiameseEfficientNet().to(device)

    # Create dummy input based on verified shapes
    dummy_target = torch.randn(2, 3, 768, 768).to(device)
    dummy_contra = torch.randn(2, 3, 768, 768).to(device)

    print("Performing Forward Pass...")
    logits = model(dummy_target, dummy_contra)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (2, 1), f"Expected output shape (2, 1), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Demonstrating Training Loop...")

    # Setup for training loop
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # We need to re-patch the dataset for the training loop execution
    with patch.object(
        SiameseMammographyDataset,
        "_load_image",
        side_effect=mock_load_image,
        autospec=True,
    ):

        # Re-initialize loaders inside the patch context to ensure they use the mocked method
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

        print(f"Running 1 Epoch of Training on {Config.DEVICE}...")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=1
        )
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        print("Running Validation...")
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val pF1:  {val_pf1:.6f}")

        # Check assertions
        assert train_loss >= 0, "Training loss should be non-negative."
        assert val_loss >= 0, "Validation loss should be non-negative."
        assert 0 <= val_pf1 <= 1, "pF1 score should be between 0 and 1."

    print("Training loop verification passed.")

    print("\n" + "=" * 50)
    print("Demonstration Completed Successfully")
    print("=" * 50)


if __name__ == "__main__":
    run_demonstration()
