import torch
import numpy as np
import os
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, calc_iou
from library.data import get_loaders
from library.model import SaltUNetPlusPlus
from library.losses import get_loss
from library.engine import train_one_epoch, validate_one_epoch


def run_demo():
    print("=== Salt Segmentation Task Demo ===")

    # 1. Setup and Configuration
    # Override Config for a fast demonstration
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = (
        0  # Use 0 for simple debugging/demo to avoid multiprocessing overhead
    )
    Config.EPOCHS = 2  # Minimal epochs
    Config.PHASE1_EPOCHS = 1

    # Ensure working directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Verify Utilities (RLE and Metrics)
    print("\n[1/4] Verifying Utilities...")

    # Create a synthetic 101x101 mask with a 10x10 square of salt
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    # Test RLE Encoding and Decoding
    encoded_rle = rle_encode(dummy_mask)
    decoded_mask = rle_decode(encoded_rle, (101, 101))

    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError(
            "RLE Encode/Decode cycle failed: Decoded mask does not match original."
        )
    print(" - RLE Encode/Decode: PASS")

    # Test IoU Calculation
    iou_perfect = calc_iou(dummy_mask, dummy_mask)
    if not np.isclose(iou_perfect, 1.0):
        raise AssertionError(
            f"IoU for identical masks should be 1.0, got {iou_perfect}"
        )

    empty_mask = np.zeros((101, 101), dtype=np.uint8)
    iou_zero = calc_iou(dummy_mask, empty_mask)
    if not np.isclose(iou_zero, 0.0):
        raise AssertionError(f"IoU for disjoint masks should be 0.0, got {iou_zero}")
    print(" - IoU Calculation: PASS")

    # 3. Verify Data Loading
    print("\n[2/4] Verifying Data Loading...")

    # Load data in debug mode (small subset)
    # fold_idx=0 implies we are using the first fold of the stratified split
    train_loader, val_loader = get_loaders(fold_idx=0, debug=True)

    # Fetch a single batch
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader returned no data.")

    print(f" - Batch Shapes -> Inputs: {inputs.shape}, Targets: {targets.shape}")

    # Assertions for shapes
    # Expected Input: (B, 3, 128, 128) -> 3 channels (Seismic, Seismic, Depth), padded to 128
    # Expected Target: (B, 1, 128, 128)
    if inputs.shape != (Config.BATCH_SIZE, 3, 128, 128):
        raise AssertionError(
            f"Expected input shape ({Config.BATCH_SIZE}, 3, 128, 128), got {inputs.shape}"
        )

    if targets.shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(
            f"Expected target shape ({Config.BATCH_SIZE}, 1, 128, 128), got {targets.shape}"
        )

    # Verify Depth Channel (Channel index 2)
    # The depth channel should be spatially constant for a given image
    depth_slice = inputs[0, 2, :, :]
    if torch.std(depth_slice) > 1e-5:
        raise AssertionError("Depth channel is not spatially constant.")
    print(" - Data Shapes & Content: PASS")

    # 4. Verify Model Architecture
    print("\n[3/4] Verifying Model Architecture...")

    model = SaltUNetPlusPlus(deep_supervision=True)
    model.to(device)
    inputs = inputs.to(device)

    # Test Forward Pass with Deep Supervision (Phase 1 mode)
    outputs_ds = model(inputs, deep_supervision=True)

    if not isinstance(outputs_ds, list):
        raise AssertionError("Model with deep_supervision=True should return a list.")
    if len(outputs_ds) != 4:
        raise AssertionError(
            f"Expected 4 outputs for deep supervision, got {len(outputs_ds)}"
        )
    if outputs_ds[-1].shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(f"Final output shape mismatch. Got {outputs_ds[-1].shape}")
    print(" - Deep Supervision Forward Pass: PASS")

    # Test Forward Pass without Deep Supervision (Phase 2 mode)
    outputs_std = model(inputs, deep_supervision=False)

    if not isinstance(outputs_std, torch.Tensor):
        raise AssertionError(
            "Model with deep_supervision=False should return a Tensor."
        )
    if outputs_std.shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(f"Output shape mismatch. Got {outputs_std.shape}")
    print(" - Standard Forward Pass: PASS")

    # 5. Verify Training Loop (Engine)
    print("\n[4/4] Verifying Training & Validation Loop...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
    )

    # Handle Scaler based on device availability
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Simulate Phase 1: Structural Warm-up ---
    print(" -> Phase 1: BCE + Dice Loss (Deep Supervision)")
    loss_fn_p1 = get_loss("phase1")

    # Train 1 Epoch
    train_loss_p1 = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        loss_fn_p1,
        device,
        epoch=1,
        phase_name="phase1",
    )
    print(f"    Train Loss: {train_loss_p1:.4f}")

    # Validate 1 Epoch
    val_loss_p1, val_map_p1 = validate_one_epoch(
        model, val_loader, loss_fn_p1, device, phase_name="phase1"
    )
    # Note: mAP might be 0.0 initially if the model hasn't learned anything yet
    print(f"    Val Loss: {val_loss_p1:.4f} | mAP: {val_map_p1:.4f}")

    # --- Simulate Phase 2: Metric Fine-tuning ---
    print(" -> Phase 2: Lovasz-Hinge Loss (Standard Output)")
    loss_fn_p2 = get_loss("phase2")

    # Train 1 Epoch
    train_loss_p2 = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        loss_fn_p2,
        device,
        epoch=2,
        phase_name="phase2",
    )
    print(f"    Train Loss: {train_loss_p2:.4f}")

    # Validate 1 Epoch
    val_loss_p2, val_map_p2 = validate_one_epoch(
        model, val_loader, loss_fn_p2, device, phase_name="phase2"
    )
    print(f"    Val Loss: {val_loss_p2:.4f} | mAP: {val_map_p2:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
