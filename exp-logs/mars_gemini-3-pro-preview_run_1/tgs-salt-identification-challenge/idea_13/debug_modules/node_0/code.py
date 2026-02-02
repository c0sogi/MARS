import os
import sys
import numpy as np
import torch
import cv2
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, compute_map_score
from library.dataset import get_dataloaders
from library.model import DeepResUNet
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Salt Segmentation Task Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Patch Config to run a very short, small experiment
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.CYCLES = 1  # 1 Cycle
    Config.CYCLE_LEN = 2  # Cycle length matches epochs
    Config.SNAPSHOT_CYCLES = [1]  # Save snapshot at end of cycle 1
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce worker overhead
    Config.LOSS_SWITCH_EPOCH = 1  # Switch loss after 1 epoch to test both phases

    # Ensure reproducibility
    set_seed(Config.SEED)
    print("Configuration patched: DEBUG=True, EPOCHS=2, BATCH_SIZE=8")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utility Functions (RLE & mAP)...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a 10x10 square of salt

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    if not np.array_equal(dummy_mask, decoded):
        raise AssertionError("RLE Encoding/Decoding failed validation.")
    print("RLE Encoding/Decoding: OK")

    # Test mAP Score
    # Case 1: Perfect match
    score_perfect = compute_map_score(dummy_mask[None, ...], dummy_mask[None, ...])
    if not np.isclose(score_perfect, 1.0):
        raise AssertionError(
            f"mAP for perfect match should be 1.0, got {score_perfect}"
        )

    # Case 2: No overlap (Pred is empty, Target is not)
    empty_pred = np.zeros_like(dummy_mask)
    score_miss = compute_map_score(empty_pred[None, ...], dummy_mask[None, ...])
    if not np.isclose(score_miss, 0.0):
        raise AssertionError(f"mAP for total miss should be 0.0, got {score_miss}")
    print("mAP Metric Calculation: OK")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Data Loading...")

    # Get dataloaders (this will use the patched Config)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Check Shapes
    # Image: (B, 1, 128, 128) - 128 due to padding in Config
    expected_shape = (Config.BATCH_SIZE, 1, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Image batch shape mismatch. Expected {expected_shape}, got {images.shape}"
        )

    if masks.shape != expected_shape:
        raise AssertionError(
            f"Mask batch shape mismatch. Expected {expected_shape}, got {masks.shape}"
        )

    if depths.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Depth batch shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {depths.shape}"
        )

    print(
        f"Batch Shapes Verified: Images {images.shape}, Masks {masks.shape}, Depths {depths.shape}"
    )

    # -------------------------------------------------------------------------
    # 4. Verify Model and Loss
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture and Loss Functions...")

    device = torch.device(Config.DEVICE)
    model = DeepResUNet().to(device)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Forward Pass (Training Mode -> Returns Deep Supervision)
    model.train()
    logits, aux2, aux1 = model(images, depths)

    if logits.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {logits.shape}"
        )
    print("Model Forward Pass (Train): OK")

    # Test Losses
    criterion_bce = BCEDiceLoss()
    criterion_lovasz = LovaszHingeLoss()

    loss_1 = criterion_bce(logits, masks)
    loss_2 = criterion_lovasz(logits, masks)

    if torch.isnan(loss_1) or loss_1.item() < 0:
        raise AssertionError("BCEDiceLoss returned NaN or negative value.")
    if torch.isnan(loss_2) or loss_2.item() < 0:
        raise AssertionError("LovaszHingeLoss returned NaN or negative value.")

    print(
        f"Loss Calculation: OK (BCE+Dice: {loss_1.item():.4f}, Lovasz: {loss_2.item():.4f})"
    )

    # -------------------------------------------------------------------------
    # 5. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 5] Executing Training Loop (Trainer.fit)...")

    # Instantiate Trainer
    trainer = Trainer()

    # Run fit (This will run for 2 epochs as configured)
    trainer.fit()

    # Verify Checkpoints exist
    expected_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(expected_ckpt):
        raise FileNotFoundError("Training finished but 'best_model.pth' was not found.")
    print("Training Loop Completed. Checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Inference Demo
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Inference on Test Batch...")

    # Load best model
    model.load_state_dict(torch.load(expected_ckpt, map_location=device))
    model.eval()

    # Get test batch
    test_images, test_depths, test_ids = next(iter(test_loader))
    test_images = test_images.to(device)
    test_depths = test_depths.to(device)

    with torch.no_grad():
        # Inference returns only logits in eval mode
        test_logits = model(test_images, test_depths)
        test_preds = torch.sigmoid(test_logits)

    # Crop back to original size (101x101)
    start_idx = (Config.IMG_HEIGHT - Config.ORIG_HEIGHT) // 2
    end_idx = start_idx + Config.ORIG_HEIGHT
    test_preds_cropped = test_preds[:, :, start_idx:end_idx, start_idx:end_idx]

    # Convert to binary mask
    pred_mask = (test_preds_cropped > 0.5).float().cpu().numpy()

    # Generate RLE for first image in batch
    sample_rle = rle_encode(pred_mask[0, 0])
    sample_id = test_ids[0]

    print(f"Inference Successful.")
    print(f"Sample ID: {sample_id}")
    print(f"Generated RLE (first 50 chars): {sample_rle[:50]}...")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
