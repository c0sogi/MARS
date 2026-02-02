import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from unittest.mock import MagicMock

# Import from the provided library files
from library.config import Config, setup_reproducibility
from library.data import get_dataloaders
from library.model import get_model
from library.train import train_one_epoch, validate, DiceLoss
from library.utils import fbeta_score, rle_encoding
import library.inference  # Import module to patch functions for demo


def main():
    print("=== Vesuvius Ink Detection: Code Demonstration ===")

    # 1. Setup and Configuration
    # Override Config for speed
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    # Ensure working dir exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    setup_reproducibility(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\n--- Testing Data Loading ---")
    # get_dataloaders handles caching and dataset creation
    train_loader, val_loader, test_loader, test_df = get_dataloaders(
        load_cached_data=True
    )

    # Verify Train Loader
    try:
        images, masks = next(iter(train_loader))
        print(f"Train Batch - Images Shape: {images.shape}, Masks Shape: {masks.shape}")

        # Assertions
        # Expected: (B, 6, 512, 512) for images, (B, 1, 512, 512) for masks
        assert images.shape == (
            Config.BATCH_SIZE,
            Config.IN_CHANNELS,
            Config.TILE_SIZE,
            Config.TILE_SIZE,
        ), f"Incorrect image shape: {images.shape}"
        assert masks.shape == (
            Config.BATCH_SIZE,
            1,
            Config.TILE_SIZE,
            Config.TILE_SIZE,
        ), f"Incorrect mask shape: {masks.shape}"
        assert images.dtype == torch.float32, "Images should be float32"
        # Check normalization (should be roughly 0-1)
        assert (
            images.max() <= 1.0 and images.min() >= 0.0
        ), "Images not normalized to [0,1]"
        print("Data loading verification passed.")
    except StopIteration:
        print("Error: Train loader is empty.")
        sys.exit(1)

    # 3. Model Initialization
    print("\n--- Testing Model Initialization ---")
    model = get_model()
    model.to(device)

    # Verify input channels modification
    # The first layer is model.segformer.encoder.patch_embeddings[0].proj
    first_layer = model.segformer.encoder.patch_embeddings[0].proj
    print(f"First Layer Input Channels: {first_layer.in_channels}")

    assert (
        first_layer.in_channels == Config.IN_CHANNELS
    ), f"Model input channels expected {Config.IN_CHANNELS}, got {first_layer.in_channels}"
    print("Model structure verification passed.")

    # 4. Training and Validation Loop
    print("\n--- Testing Training and Validation Steps ---")

    # Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Define Criterion (BCE + Dice)
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()

    def criterion(preds, targets):
        bce = bce_fn(preds, targets)
        dice = dice_fn(preds, targets)
        return Config.BCE_WEIGHT * bce + Config.DICE_WEIGHT * dice

    # Run one training epoch
    # Since DEBUG is True, the loader has very few batches, so this is fast.
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"

    # Run validation
    print("Running validate...")
    val_loss, val_score = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val F0.5 Score: {val_score:.4f}")
    assert 0.0 <= val_score <= 1.0, "F0.5 score should be between 0 and 1"

    # 5. Utilities Verification
    print("\n--- Testing Utilities ---")

    # Test fbeta_score
    # Perfect match
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred = torch.tensor([0.9, 0.1, 0.9, 0.1])  # Probabilities
    score_perfect = fbeta_score(y_pred, y_true, threshold=0.5, beta=0.5)
    print(f"F0.5 Score (Perfect Match): {score_perfect:.4f}")
    assert np.isclose(
        score_perfect, 1.0, atol=1e-4
    ), "Fbeta score calculation failed for perfect match"

    # Test RLE Encoding
    # Mask: 0 1 1 0
    # Pixels (1-based): 1(0), 2(1), 3(1), 4(0)
    # Ink at 2, 3. Start 2, Length 2.
    dummy_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # Flattened: 0, 1, 1, 0
    rle_str = rle_encoding(dummy_mask)
    print(f"RLE Output: '{rle_str}'")
    assert rle_str == "2 2", f"RLE encoding incorrect. Expected '2 2', got '{rle_str}'"

    # 6. Inference Pipeline Verification
    print("\n--- Testing Inference Pipeline (Mocked) ---")

    # To demonstrate predict_fragment without processing a massive file,
    # we mock the data loading functions in library.inference.

    # Mock data loader to return a small random volume (6, 512, 512)
    # This simulates loading a cached MIP file.
    def mock_load_mips(fid, path, load_cached_data=True):
        print(f"  [Mock] Loading MIPs for fragment {fid}...")
        return np.random.randint(
            0, 65535, (Config.IN_CHANNELS, 512, 512), dtype=np.uint16
        )

    # Mock cv2.imread to return a small mask (512, 512) matching the volume
    original_imread = cv2.imread

    def mock_imread(path, flags=None):
        if "mask.png" in path:
            print(f"  [Mock] Loading mask from {path}...")
            return np.ones((512, 512), dtype=np.uint8) * 255
        return original_imread(path, flags)

    # Apply mocks
    library.inference.load_fragment_mips = mock_load_mips
    library.inference.cv2.imread = mock_imread

    # Run prediction on a dummy fragment ID
    # This exercises the sliding window, TTA, normalization, and thresholding logic in predict_fragment.
    try:
        pred_mask = library.inference.predict_fragment(
            model,
            fragment_id="test_frag",
            volume_path="dummy/path",
            mask_path="dummy/mask.png",
            device=device,
            tta=True,  # Test with TTA enabled
        )

        print(f"Prediction Output Shape: {pred_mask.shape}")
        assert pred_mask.shape == (512, 512), "Prediction mask shape mismatch"
        assert pred_mask.dtype == np.uint8, "Prediction mask should be uint8"
        assert set(np.unique(pred_mask)).issubset(
            {0, 1}
        ), "Prediction mask should be binary"
        print("Inference pipeline verification passed.")

    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
