import os
import shutil
import numpy as np
import torch
import pandas as pd
import cv2

# Import from the provided library
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    do_kaggle_metric,
)
from library.dataset import get_dataloaders
from library.model import HyperResUNet
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.train import train_model


def run_demo():
    print("=== Starting Salt Segmentation Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Set a specific working directory for this demo to avoid conflicts
    demo_working_dir = os.path.join("working", "demo_execution")
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Modify Config attributes globally
    Config.WORKING_DIR = demo_working_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small subset
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.NUM_EPOCHS = 2
    Config.PHASE_1_EPOCHS = 1  # Switch to Phase 2 after 1 epoch
    Config.CYCLE_LEN = 1

    # Update checkpoint paths in Config to point to new dir
    Config.CYCLE_2_BEST_MODEL = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_2.pth")
    Config.CYCLE_3_BEST_MODEL = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_3.pth")

    Config.set_seed(42)
    print("    Configuration updated: DEBUG=True, Epochs=2, Batch=4")

    # -------------------------------------------------------------------------
    # 2. Validate Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Validating Utility Functions...")

    # A. RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(dummy_mask, decoded), "RLE decode did not match original mask"
    print("    RLE Encoding/Decoding: PASSED")

    # B. Padding/Unpadding
    dummy_img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded_img = pad_image(dummy_img)

    assert padded_img.shape == (128, 128), f"Padded shape mismatch: {padded_img.shape}"

    unpadded_img = unpad_image(padded_img, original_shape=(101, 101))
    assert np.array_equal(
        dummy_img, unpadded_img
    ), "Unpadding did not restore original image"
    print("    Image Padding/Unpadding: PASSED")

    # C. Kaggle Metric (mAP at IoU thresholds)
    # Case 1: Perfect match
    score_perfect = do_kaggle_metric(dummy_mask[None, ...], dummy_mask[None, ...])
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match score should be 1.0, got {score_perfect}"

    # Case 2: No overlap
    empty_mask = np.zeros_like(dummy_mask)
    score_zero = do_kaggle_metric(dummy_mask[None, ...], empty_mask[None, ...])
    assert np.isclose(
        score_zero, 0.0
    ), f"No overlap score should be 0.0, got {score_zero}"
    print("    Kaggle Metric Calculation: PASSED")

    # -------------------------------------------------------------------------
    # 3. Validate Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Validating Data Loading...")

    # Force reload to ensure debug sampling works
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True, batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Check shapes: (B, C, H, W)
    # Config.INPUT_CHANNELS = 2 (Image + Depth)
    # Config.INPUT_HEIGHT/WIDTH = 128
    assert images.shape == (
        Config.BATCH_SIZE,
        2,
        128,
        128,
    ), f"Image batch shape incorrect: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Mask batch shape incorrect: {masks.shape}"

    # Check value ranges
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images not normalized to [0, 1]"
    assert set(np.unique(masks.numpy())).issubset({0, 1}), "Masks are not binary"

    print(
        f"    Batch loaded successfully. Images: {images.shape}, Masks: {masks.shape}"
    )
    print("    Data Loading: PASSED")

    # -------------------------------------------------------------------------
    # 4. Validate Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Validating Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = HyperResUNet().to(device)
    model.train()  # Enable Deep Supervision

    # Forward pass
    outputs = model(images.to(device))

    # Expecting a list of outputs because Deep Supervision is ON
    assert isinstance(
        outputs, list
    ), "Model in train mode should return a list of outputs"
    assert len(outputs) > 1, "Deep supervision should produce multiple outputs"
    assert outputs[0].shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Main output shape mismatch"

    print(
        f"    Forward pass successful. Generated {len(outputs)} outputs (Deep Supervision)."
    )
    print("    Model Architecture: PASSED")

    # -------------------------------------------------------------------------
    # 5. Validate Loss Functions
    # -------------------------------------------------------------------------
    print("\n[5] Validating Loss Functions...")

    # Phase 1 Loss: BCE + Dice
    criterion_p1 = BCEDiceLoss()
    loss_p1 = criterion_p1(outputs[0], masks.to(device))
    assert not torch.isnan(loss_p1), "BCE+Dice Loss returned NaN"
    assert loss_p1.item() > 0, "BCE+Dice Loss should be positive"

    # Phase 2 Loss: BCE + Lovasz
    criterion_p2 = LovaszHingeLoss()
    # Lovasz expects logits, model returns logits (no sigmoid applied at end of model)
    loss_p2 = criterion_p2(outputs[0], masks.to(device))
    assert not torch.isnan(loss_p2), "Lovasz Loss returned NaN"

    print(f"    BCE+Dice Loss: {loss_p1.item():.4f}")
    print(f"    Lovasz Loss: {loss_p2.item():.4f}")
    print("    Loss Functions: PASSED")

    # -------------------------------------------------------------------------
    # 6. Execute Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[6] Executing Training Loop (2 Epochs)...")

    # We use the provided train_model function which instantiates Trainer
    # and runs the fit method. Since we patched Config, it uses our settings.

    try:
        train_model()
        print("    Training loop completed successfully.")
    except Exception as e:
        print(f"    Training loop failed with error: {e}")
        raise e

    # Verify output files exist
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print(f"    Checkpoint verified at: {best_model_path}")
    print("    Training Execution: PASSED")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
