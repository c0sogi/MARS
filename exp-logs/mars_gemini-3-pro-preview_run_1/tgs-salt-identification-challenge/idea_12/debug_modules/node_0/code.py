import os
import shutil
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    compute_iou_batch,
    compute_map_batch,
)
from library.dataset import get_dataloaders
from library.model_components import SaltUNet
from library.losses import ConsistentCompoundLoss
from library.engine import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Salt Segmentation Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("1. Setting up Configuration...")

    # Override Config parameters for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.CYCLES = 1
    Config.EPOCHS_PER_CYCLE = 1

    # Use a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_submission")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.PREDICTIONS_DIR = os.path.join(Config.WORKING_DIR, "predictions")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Validating Utility Functions
    # -------------------------------------------------------------------------
    print("\n2. Validating Utility Functions (RLE & Metrics)...")

    # Test RLE Encoding and Decoding
    # Create a synthetic mask: 101x101 with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    # Check if decoded matches original
    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE Decode does not match original mask!"
    print("   [Pass] RLE Encode/Decode consistency check.")

    # Test Metric Calculation (IoU and mAP)
    # Case 1: Perfect match
    pred_perfect = np.zeros((1, 101, 101), dtype=np.uint8)
    pred_perfect[0, 10:20, 10:20] = 1
    label_perfect = pred_perfect.copy()

    iou_perfect = compute_iou_batch(pred_perfect, label_perfect)
    assert np.isclose(iou_perfect[0], 1.0), f"Expected IoU 1.0, got {iou_perfect[0]}"

    map_perfect = compute_map_batch(pred_perfect, label_perfect, thresholds=[0.5, 0.95])
    assert np.isclose(map_perfect, 1.0), f"Expected mAP 1.0, got {map_perfect}"

    # Case 2: No overlap
    pred_empty = np.zeros((1, 101, 101), dtype=np.uint8)
    iou_zero = compute_iou_batch(pred_empty, label_perfect)
    assert np.isclose(iou_zero[0], 0.0), f"Expected IoU 0.0, got {iou_zero[0]}"

    print("   [Pass] Metric calculation (IoU & mAP) check.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n3. Initializing Data Loaders...")

    # This will load metadata from ./metadata/*.csv and images from ./input
    # It will cache processed arrays to ./working/demo_execution/cache
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=False
    )

    print(f"   Train Batches: {len(train_loader)}")
    print(f"   Val Batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, masks, depths, ids = next(iter(train_loader))

    # Expected shapes:
    # Images: (B, 1, 128, 128) - padded from 101
    # Masks: (B, 1, 128, 128)
    # Depths: (B,)
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected mask shape: {masks.shape}"
    assert (
        depths.shape[0] == Config.BATCH_SIZE
    ), f"Unexpected depths shape: {depths.shape}"

    print(f"   [Pass] Batch shapes verified: Img {images.shape}, Mask {masks.shape}")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n4. Initializing SaltUNet Model...")

    device = Config.DEVICE
    model = SaltUNet().to(device)

    # Verify forward pass
    images = images.to(device)
    depths = depths.to(device)

    # Forward pass (training mode, deep supervision enabled by default in Config)
    model.train()
    outputs = model(images, depths)

    # Should return list [logits, aux_64, aux_32]
    assert isinstance(
        outputs, list
    ), "Model should return a list in training mode with deep supervision."
    assert len(outputs) == 3, f"Expected 3 outputs (main + 2 aux), got {len(outputs)}"
    assert outputs[0].shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Main output shape mismatch: {outputs[0].shape}"

    print("   [Pass] Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n5. Computing Loss...")

    criterion = ConsistentCompoundLoss().to(device)
    masks = masks.to(device)

    # Calculate loss on the batch
    # We need to manually handle the list output for the loss function check here
    # (similar to how engine.py does it)
    logits = outputs[0]
    loss = criterion(logits, masks)

    # Aux losses
    for aux in outputs[1:]:
        aux_upsampled = torch.nn.functional.interpolate(
            aux, size=masks.shape[2:], mode="bilinear", align_corners=True
        )
        loss += 0.5 * criterion(aux_upsampled, masks)

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive."

    print(f"   [Pass] Loss computed successfully: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 6. Training & Validation Loop
    # -------------------------------------------------------------------------
    print("\n6. Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run training for one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"   Epoch 1 Train Loss: {train_loss:.4f}")

    # Run validation
    print("   Running Validation...")
    val_map = validate(model, val_loader, device)
    print(f"   Epoch 1 Val mAP: {val_map:.4f}")

    # Save checkpoint
    torch.save(
        model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    )
    print("   [Pass] Training and Validation cycle complete.")

    # -------------------------------------------------------------------------
    # 7. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n7. Running Inference on Test Set...")

    # Switch to eval mode
    model.eval()
    submission_data = []

    # Iterate through test loader (subset)
    with torch.no_grad():
        for images, depths, ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # Forward pass
            out = model(images, depths)

            # In eval mode, SaltUNet might return list or tensor depending on logic.
            # Based on source code: if self.training and self.deep_supervision -> list.
            # So in eval, it returns tensor logits.
            if isinstance(out, list):
                out = out[0]

            probs = torch.sigmoid(out)

            # Crop back to 101x101
            # Current shape: (B, 1, 128, 128)
            h, w = probs.shape[2], probs.shape[3]
            orig_h, orig_w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE
            start_h = (h - orig_h) // 2
            start_w = (w - orig_w) // 2

            probs = probs[:, :, start_h : start_h + orig_h, start_w : start_w + orig_w]

            # Threshold
            preds = (probs > 0.5).cpu().numpy()

            # Encode
            for i in range(len(ids)):
                pred_mask = preds[i, 0]
                rle = rle_encode(pred_mask)
                submission_data.append([ids[i], rle])

    # Save submission
    sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"   Submission saved to: {sub_path}")
    print(f"   Generated {len(sub_df)} predictions.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
