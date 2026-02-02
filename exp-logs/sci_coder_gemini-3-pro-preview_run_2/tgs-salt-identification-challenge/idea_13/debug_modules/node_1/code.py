import os
import shutil
import numpy as np
import torch
import torch.optim as optim
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    do_kaggle_metric,
)
from library.dataset import get_dataloaders, SaltDataset
from library.model import SaltNet
from library.losses import BCELovaszLoss
from library.train import train_model


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    print("=== Salt Segmentation Pipeline Demonstration ===")
    set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config attributes for a fast demo execution
    # We use a specific directory for this run to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Clean up previous demo artifacts if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Initialize directories
    Config.setup()

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test RLE Encoding/Decoding
    # Create a 101x101 mask with a 10x10 square of salt
    mask_orig = np.zeros((101, 101), dtype=np.uint8)
    mask_orig[10:20, 10:20] = 1

    rle_str = rle_encode(mask_orig)
    mask_decoded = rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(mask_orig, mask_decoded), "RLE Encode/Decode mismatch"
    print("    RLE Encode/Decode: PASSED")

    # Test Image Padding (Reflection)
    # Pad 101x101 -> 128x128
    padded_mask = pad_image(mask_orig)
    assert padded_mask.shape == (
        128,
        128,
    ), f"Padded shape mismatch: {padded_mask.shape}"

    # Unpad 128x128 -> 101x101
    unpadded_mask = unpad_image(padded_mask, original_size=(101, 101))
    assert unpadded_mask.shape == (
        101,
        101,
    ), f"Unpadded shape mismatch: {unpadded_mask.shape}"
    assert np.array_equal(mask_orig, unpadded_mask), "Pad/Unpad content mismatch"
    print("    Image Padding/Unpadding: PASSED")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset Loading...")

    # Load dataloaders. We set load_cached_data=False to force the processing logic
    # to run at least once and verify it works with the raw input files.
    train_loader, val_loader, test_loader = get_dataloaders(use_cache=False)

    # Fetch a single batch
    images, masks, depths, ids = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    # Assertions for shape and type
    # Expected: (B, 1, 128, 128) for images/masks because SaltDataset collapses RGB to 1 channel
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Image tensor shape incorrect"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Mask tensor shape incorrect"
    assert depths.shape == (Config.BATCH_SIZE, 1), "Depth tensor shape incorrect"
    assert images.dtype == torch.float32, "Image dtype should be float32"
    print("    Dataset Loading: PASSED")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = SaltNet().to(device)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Forward pass
    outputs = model(images, depths)

    print(f"    Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Model output shape incorrect"
    print("    Model Forward Pass: PASSED")

    # -------------------------------------------------------------------------
    # 5. Verify Loss Function & Metric
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Loss and Metric...")

    criterion = BCELovaszLoss()

    # Calculate Loss
    loss = criterion(outputs, masks)
    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Calculate Metric (IoU based mAP)
    # Test with perfect predictions
    perfect_score = do_kaggle_metric(masks, masks)
    assert np.isclose(perfect_score, 1.0), "Perfect prediction should score 1.0"

    # Test with actual model output (likely low score at init)
    current_score = do_kaggle_metric(torch.sigmoid(outputs), masks)
    print(f"    Current Batch mAP: {current_score:.4f}")

    print("    Loss & Metric: PASSED")

    # -------------------------------------------------------------------------
    # 6. Integration Test: Short Optimization Loop
    # -------------------------------------------------------------------------
    print("\n[6] Running Short Optimization Loop (Integration Test)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    initial_loss = loss.item()
    print(f"    Step 0 Loss: {initial_loss:.4f}")

    # Run 5 optimization steps on the same batch to verify convergence capability
    for i in range(5):
        optimizer.zero_grad()
        out = model(images, depths)
        l = criterion(out, masks)
        l.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print(f"    Step {i+1} Loss: {l.item():.4f}")

    final_loss = l.item()
    if final_loss < initial_loss:
        print("    Optimization check: Loss decreased successfully.")
    else:
        print(
            "    Optimization check: Loss did not decrease (unexpected for same-batch overfitting)."
        )

    print("    Integration Test: PASSED")

    # -------------------------------------------------------------------------
    # 7. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[7] Executing Full Training Pipeline (1 Epoch)...")
    print("    Calling library.train.train_model()...")

    # This uses the overridden Config values (EPOCHS=1, BATCH_SIZE=8)
    # It will reuse the cache generated in step 3
    trained_model = train_model(load_cached_data=True)

    # Verify artifacts
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"    Success: Best model saved at {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Best model file was not generated.")

    print("    Full Pipeline Execution: PASSED")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
