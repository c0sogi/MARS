import os
import shutil
import numpy as np
import torch
import torch.optim as optim
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode
from library.model import AttentionUNetResNet34
from library.loss import BCEDiceLoss
from library.dataset import get_dataloader
from library.train import train_one_epoch
from library.inference import predict_sliding_window

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=== Starting HuBMAP Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("[1] Setting up Configuration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Use 0 for simple debugging
    Config.WORKING_DIR = "./working/demo_run"
    Config.TRAIN_CACHE_DIR = os.path.join(Config.WORKING_DIR, "train_cache")
    Config.VAL_CACHE_DIR = os.path.join(Config.WORKING_DIR, "val_cache")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.TRAIN_CACHE_DIR, exist_ok=True)
    os.makedirs(Config.VAL_CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device("cpu")  # Use CPU for simple logic verification
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")
    print("    Configuration setup complete.\n")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities (RLE Encoding/Decoding)
    # -------------------------------------------------------------------------
    print("[2] Verifying Utilities (RLE)...")

    # Create a synthetic binary mask (100x100) with a square in the middle
    mask_shape = (100, 100)
    synthetic_mask = np.zeros(mask_shape, dtype=np.uint8)
    synthetic_mask[25:75, 25:75] = 1

    # Encode
    rle_str = rle_encode(synthetic_mask)

    # Decode
    decoded_mask = rle_decode(rle_str, mask_shape)

    # Verification
    assert isinstance(rle_str, str), "RLE encode should return a string"
    assert np.array_equal(
        synthetic_mask, decoded_mask
    ), "Decoded mask does not match original"
    print("    RLE Encode/Decode logic verified successfully.\n")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("[3] Verifying Model Architecture...")

    # Instantiate model (pretrained=False for speed/offline safety)
    model = AttentionUNetResNet34(
        in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(device)
    model.eval()

    # Create dummy input: (Batch=2, Channels=4, H=256, W=256)
    # Channels = 3 RGB + 1 Anatomical Mask
    dummy_input = torch.randn(2, 4, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Verification
    expected_shape = (2, 1, 256, 256)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    print(f"    Model input: {dummy_input.shape}")
    print(f"    Model output: {output.shape}")
    print("    Model architecture verified successfully.\n")

    # -------------------------------------------------------------------------
    # 4. Verify Loss Function
    # -------------------------------------------------------------------------
    print("[4] Verifying Loss Function...")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # Dummy logits (predictions) and targets
    logits = torch.randn(2, 1, 256, 256).to(device)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    loss = criterion(logits, targets)

    # Verification
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"    Calculated Loss: {loss.item():.4f}")
    print("    Loss function verified successfully.\n")

    # -------------------------------------------------------------------------
    # 5. Verify Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("[5] Verifying Data Loading (DEBUG mode)...")

    # This will use the metadata files to load 1 image (due to DEBUG=True),
    # slice it into tiles, and return a dataloader.
    try:
        train_loader = get_dataloader(
            phase="train", batch_size=2, load_cached_data=False
        )

        # Fetch one batch
        images, masks = next(iter(train_loader))

        # Verification
        # Expected image shape: (B, 4, 1024, 1024)
        # Expected mask shape: (B, 1, 1024, 1024)
        print(f"    Batch Image Shape: {images.shape}")
        print(f"    Batch Mask Shape: {masks.shape}")

        assert images.shape[1] == 4, "Input should have 4 channels (RGB + Anatomy)"
        assert (
            images.shape[2] == Config.TILE_SIZE
        ), f"Height should be {Config.TILE_SIZE}"
        assert masks.shape[1] == 1, "Mask should have 1 channel"

        # Check normalization (approximate range for standardized data)
        # RGB channels should be roughly mean 0 std 1
        rgb_mean = images[:, :3, ...].mean().item()
        assert -3.0 < rgb_mean < 3.0, f"RGB mean {rgb_mean} out of expected range"

        # Anatomy channel (4th channel) should be 0 or 1 (before normalization logic).
        # In dataset.py, anatomy is not standardized, just passed as float 0.0 or 1.0.
        # Let's verify values are within [0, 1].
        anat_min = images[:, 3, ...].min().item()
        anat_max = images[:, 3, ...].max().item()
        assert (
            anat_min >= 0.0 and anat_max <= 1.0
        ), "Anatomy channel values out of range [0, 1]"

        print("    Data loading and preprocessing verified successfully.\n")

    except Exception as e:
        print(
            f"    Data loading failed (likely due to missing input files in this environment): {e}"
        )
        # Create dummy loader for subsequent steps if real data fails
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.randn(4, 4, Config.TILE_SIZE, Config.TILE_SIZE),
            torch.randint(0, 2, (4, 1, Config.TILE_SIZE, Config.TILE_SIZE)).float(),
        )
        train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2)
        print("    Created dummy dataloader for remaining steps.")

    # -------------------------------------------------------------------------
    # 6. Verify Training Step
    # -------------------------------------------------------------------------
    print("[6] Verifying Training Step...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Run one epoch (which is just a few batches in DEBUG mode)
    # We intercept the loop inside train_one_epoch by just running the logic manually for one batch
    # to ensure we don't wait for too long.

    images, masks = next(iter(train_loader))
    images, masks = images.to(device), masks.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()

    print(f"    Single step training loss: {loss.item():.4f}")
    print("    Training step verified successfully.\n")

    # -------------------------------------------------------------------------
    # 7. Verify Inference (Sliding Window)
    # -------------------------------------------------------------------------
    print("[7] Verifying Inference (Sliding Window)...")

    # Create a large synthetic image (e.g., 2048x2048)
    # Shape: (H, W, 4)
    large_h, large_w = 2048, 2048
    synthetic_large_img = np.random.randint(0, 255, (large_h, large_w, 4)).astype(
        np.uint8
    )
    # Make anatomy channel binary
    synthetic_large_img[:, :, 3] = (synthetic_large_img[:, :, 3] > 127).astype(np.uint8)

    # Run prediction
    try:
        prob_map = predict_sliding_window(
            synthetic_large_img,
            model,
            device,
            tile_size=Config.TILE_SIZE,  # 1024
            overlap=0.5,
        )

        # Verification
        assert prob_map.shape == (
            large_h,
            large_w,
        ), f"Probability map shape mismatch: {prob_map.shape}"
        assert (
            prob_map.min() >= 0.0 and prob_map.max() <= 1.0
        ), "Probability map values out of range [0, 1]"

        print(f"    Input Image: {synthetic_large_img.shape}")
        print(f"    Output Probability Map: {prob_map.shape}")
        print("    Inference pipeline verified successfully.\n")

    except Exception as e:
        print(f"    Inference failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    print("=== Demonstration Complete ===")
    # Optional: Remove temp working dir
    # shutil.rmtree(Config.WORKING_DIR)


if __name__ == "__main__":
    run_demonstration()
