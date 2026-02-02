import os
import sys
import numpy as np
import torch
import cv2

# Import necessary components from the provided library
from library.config import Config
from library.utils import rle_encode, rle_decode, iou_metric
from library.dataset import get_dataloaders, set_seed
from library.model import ResNeXtUNet
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.train import run_training


def demonstrate_utils():
    """Verifies utility functions for RLE and Metrics."""
    print("\n=== Demonstrating Utils ===")

    # 1. RLE Encoding/Decoding
    # Create a dummy 101x101 mask
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # 10x10 square

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert np.array_equal(mask, decoded), "RLE Decode does not match original mask"
    print("RLE Encode/Decode verification passed.")

    # 2. IoU Metric
    # Perfect overlap
    iou_perfect = iou_metric(mask[None, ...], mask[None, ...])
    assert np.isclose(iou_perfect, 1.0), f"Expected IoU 1.0, got {iou_perfect}"

    # No overlap
    mask_inv = 1 - mask
    iou_zero = iou_metric(mask[None, ...], mask_inv[None, ...])
    # Note: If union is non-empty and intersection is 0, IoU is 0.
    assert np.isclose(iou_zero, 0.0), f"Expected IoU 0.0, got {iou_zero}"
    print("IoU Metric verification passed.")


def demonstrate_data_pipeline():
    """Verifies Data Loading and Preprocessing."""
    print("\n=== Demonstrating Data Pipeline ===")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 16  # Small subset
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Main process only for stability in demo

    # Force recompute cache to demonstrate processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a single batch
    inputs, masks = next(iter(train_loader))

    # Verify Shapes
    # Inputs: (Batch, 3, 128, 128) - 128 is the padded size
    assert inputs.shape == (4, 3, 128, 128), f"Unexpected input shape: {inputs.shape}"
    # Masks: (Batch, 1, 128, 128)
    assert masks.shape == (4, 1, 128, 128), f"Unexpected mask shape: {masks.shape}"

    # Verify Value Ranges
    assert (
        inputs.min() >= 0.0 and inputs.max() <= 1.0
    ), "Inputs should be normalized [0,1]"
    unique_vals = torch.unique(masks)
    assert all(v in [0, 1] for v in unique_vals), "Masks should be binary (0 or 1)"

    print("Data Loader shape and value verification passed.")
    return inputs, masks


def demonstrate_model_and_loss(inputs, masks):
    """Verifies Model Forward Pass and Loss Calculation."""
    print("\n=== Demonstrating Model and Loss ===")

    device = Config.DEVICE
    # Instantiate model (pretrained=False to speed up init for this check)
    model = ResNeXtUNet(n_classes=1, pretrained=False).to(device)

    inputs = inputs.to(device)
    masks = masks.to(device)

    # Forward Pass
    logits = model(inputs)
    assert logits.shape == masks.shape, f"Logit shape mismatch: {logits.shape}"

    # Loss Calculation: BCE + Dice
    criterion_bce = BCEDiceLoss()
    loss_bce = criterion_bce(logits, masks)
    assert not torch.isnan(loss_bce), "BCE Dice Loss is NaN"

    # Loss Calculation: Lovasz
    criterion_lovasz = LovaszHingeLoss()
    loss_lovasz = criterion_lovasz(logits, masks)
    assert not torch.isnan(loss_lovasz), "Lovasz Loss is NaN"

    print(
        f"Forward pass successful. BCE Loss: {loss_bce.item():.4f}, Lovasz Loss: {loss_lovasz.item():.4f}"
    )

    # Backward Pass Verification
    loss_bce.backward()
    print("Backward pass successful.")


def demonstrate_full_training():
    """Executes the training loop from library.train."""
    print("\n=== Demonstrating Full Training Loop ===")

    # Configure for a quick run
    Config.EPOCHS = 2
    Config.LOVASZ_EPOCH_START = 2  # Epoch 1: BCE, Epoch 2: Lovasz
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 16

    # Run the training function provided in the library
    # This handles loading data, training epochs, validation, and saving
    model, best_threshold = run_training()

    assert model is not None
    assert 0.0 < best_threshold < 1.0
    print(f"Training demo complete. Best Threshold: {best_threshold:.4f}")
    return model, best_threshold


def demonstrate_inference(model, threshold):
    """Demonstrates inference and post-processing (cropping/encoding)."""
    print("\n=== Demonstrating Inference and Post-processing ===")

    # Get test loader (using cache generated earlier)
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    model.eval()
    device = Config.DEVICE

    # Get one batch
    inputs, ids = next(iter(test_loader))
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

    preds_np = preds.cpu().numpy()  # Shape: (B, 1, 128, 128)

    # Post-processing: Crop back to original 101x101 size
    # The padding strategy in dataset.py is reflection padding to center the image.
    # Target 128, Original 101 -> Diff 27.
    # Top pad = 27 // 2 = 13.
    # Left pad = 27 // 2 = 13.
    orig_size = 101
    pad_top = (128 - 101) // 2
    pad_left = (128 - 101) // 2

    cropped_preds = preds_np[
        :, :, pad_top : pad_top + orig_size, pad_left : pad_left + orig_size
    ]

    assert cropped_preds.shape == (
        4,
        1,
        101,
        101,
    ), f"Cropping failed, shape: {cropped_preds.shape}"

    # Encode the first prediction in the batch
    mask_img = cropped_preds[0, 0].astype(np.uint8)
    rle = rle_encode(mask_img)

    print(f"Inference successful for ID: {ids[0]}")
    print(f"RLE string length: {len(rle)}")


if __name__ == "__main__":
    # Set global seed
    set_seed(42)

    # Execute demonstrations
    demonstrate_utils()
    inputs, masks = demonstrate_data_pipeline()
    demonstrate_model_and_loss(inputs, masks)
    trained_model, best_thresh = demonstrate_full_training()
    demonstrate_inference(trained_model, best_thresh)

    print("\nAll demonstrations completed successfully.")
