import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, fbeta_score, dice_coef, rle_encoding
from library.data_processing import load_fragment_slab
from library.dataset import InkDataset
from library.model import SegFormerB3
from library.training import BCEDiceLoss, train_one_epoch, validate
from library.inference import predict_fragment_scan


def run_demo():
    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("--- 1. Configuring Environment ---")
    # Override Config values to ensure the demo runs quickly
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration updated for demo mode.")

    # ==========================================
    # 2. Data Processing Demo (Slab Generation)
    # ==========================================
    print("\n--- 2. Demonstrating Data Processing ---")
    # We will load a slab for fragment '1' (Training data)
    # This tests load_volume_slices and make_overlapping_slab internally
    frag_id = "1"
    # Metadata implies volume is at train/1/surface_volume
    # We construct the relative path expected by load_fragment_slab
    # Note: load_fragment_slab expects path relative to INPUT_DIR
    volume_rel_path = f"train/{frag_id}/surface_volume"

    print(f"Loading slab for Fragment {frag_id} at Z={Config.TRAIN_Z_START}...")
    slab = load_fragment_slab(
        fragment_id=frag_id,
        volume_path=volume_rel_path,
        z_start=Config.TRAIN_Z_START,
        load_cached_data=True,
    )

    # Validation
    assert isinstance(slab, np.ndarray), "Slab must be a numpy array"
    assert slab.ndim == 3, f"Slab must be 3D (H, W, C), got {slab.ndim}"
    assert slab.shape[2] == 3, f"Slab must have 3 channels, got {slab.shape[2]}"
    assert slab.dtype == np.float32, f"Slab must be float32, got {slab.dtype}"
    assert (
        slab.min() >= 0.0 and slab.max() <= 1.0
    ), "Slab values must be normalized [0, 1]"

    print(
        f"Slab loaded successfully. Shape: {slab.shape}, Range: [{slab.min():.4f}, {slab.max():.4f}]"
    )

    # ==========================================
    # 3. Dataset & DataLoader Demo
    # ==========================================
    print("\n--- 3. Demonstrating Dataset & DataLoader ---")
    # Initialize Dataset
    # This will pre-load the slabs into RAM (might take a moment)
    train_dataset = InkDataset(mode="train")

    # Validation of dataset length
    print(f"Total training patches: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty"

    # Create a small subset for rapid iteration
    subset_indices = list(range(4))  # Only use first 4 items
    train_subset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Use 0 workers for simple demo debugging
    )

    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Validation
    print(f"Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}")
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect image tensor shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect mask tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32"

    # ==========================================
    # 4. Model Demo
    # ==========================================
    print("\n--- 4. Demonstrating Model Architecture ---")
    device = Config.DEVICE
    model = SegFormerB3().to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    logits = model(images)

    # Validation
    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Logits shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # ==========================================
    # 5. Training Loop Components Demo
    # ==========================================
    print("\n--- 5. Demonstrating Training Components ---")
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for 1 epoch on the subset
    print("Running training step (1 epoch on 4 samples)...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Validate on the subset (acting as validation set)
    print("Running validation step...")
    val_loss, val_f05, val_dice = validate(model, train_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, F0.5: {val_f05:.4f}, Dice: {val_dice:.4f}")

    # Save model for inference demo
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model checkpoint saved to {model_path}")

    # ==========================================
    # 6. Inference Demo
    # ==========================================
    print("\n--- 6. Demonstrating Inference ---")
    # We will run inference on Test Fragment 'a'
    test_frag_id = "a"
    # Note: This relies on metadata/test.csv existing and pointing to valid data

    # Check if test data exists (sanity check for the demo environment)
    if os.path.exists(os.path.join(Config.INPUT_DIR, "test", test_frag_id)):
        print(
            f"Running prediction on Test Fragment {test_frag_id} at Z={Config.INFERENCE_Z_STARTS[0]}..."
        )

        # Predict
        # We use the first Z-start from config for speed
        z_scan_depth = Config.INFERENCE_Z_STARTS[0]
        prob_map = predict_fragment_scan(model, test_frag_id, z_scan_depth, device)

        # Validation
        print(f"Prediction Map Shape: {prob_map.shape}")
        assert isinstance(prob_map, np.ndarray), "Prediction must be numpy array"
        assert prob_map.ndim == 2, "Prediction map must be 2D (H, W)"
        assert (
            prob_map.min() >= 0.0 and prob_map.max() <= 1.0
        ), "Probabilities must be [0, 1]"

        # Check against mask dimensions (read mask to verify size match)
        mask_path = os.path.join(Config.INPUT_DIR, f"test/{test_frag_id}/mask.png")
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            h, w = mask_img.shape
            assert prob_map.shape == (
                h,
                w,
            ), f"Prediction shape {prob_map.shape} does not match mask shape {(h, w)}"
    else:
        print(f"Test fragment {test_frag_id} not found. Skipping inference execution.")

    # ==========================================
    # 7. Metric / RLE Demo
    # ==========================================
    print("\n--- 7. Demonstrating RLE Encoding ---")
    # Create a synthetic binary mask: 10 pixels, ink at indices 1, 2, 3 and 8
    # 0 1 1 1 0 0 0 0 1 0
    # RLE logic:
    # 1-based indexing.
    # Run 1: Start 2, Length 3
    # Run 2: Start 9, Length 1
    # Expected: "2 3 9 1"

    dummy_mask = np.zeros((1, 10), dtype=np.uint8)
    dummy_mask[0, 1:4] = 1
    dummy_mask[0, 8] = 1

    rle_str = rle_encoding(dummy_mask)
    print(f"Mask: {dummy_mask.flatten()}")
    print(f"RLE: '{rle_str}'")

    # Validation
    expected_rle = "2 3 9 1"
    assert (
        rle_str == expected_rle
    ), f"RLE encoding incorrect. Expected '{expected_rle}', got '{rle_str}'"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
