import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import cv2

# Import from the provided library files
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    compute_map_score,
    save_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import DeepResUNet
from library.losses import BCEDiceLoss, DeepSupervisionLoss, LovaszHingeLoss
from library.engine import train_one_epoch, validate, center_crop


def main():
    print("Starting Salt Segmentation Demo...")

    # 1. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    set_seed(42)

    # Ensure working directory exists for outputs
    os.makedirs("./working/demo_checkpoints", exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n--- Verifying Utility Functions ---")

    # Test RLE Encoding/Decoding
    # Create a dummy 101x101 mask with a square in the middle
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[40:60, 40:60] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    # Check if decoded matches original
    assert np.array_equal(dummy_mask, decoded), "RLE Encode/Decode roundtrip failed!"
    print("RLE Encode/Decode verification passed.")

    # Test mAP calculation
    # Perfect match case
    score_perfect = compute_map_score(dummy_mask[None, ...], dummy_mask[None, ...])
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected mAP 1.0 for perfect match, got {score_perfect}"

    # No overlap case
    dummy_pred_empty = np.zeros_like(dummy_mask)
    # Note: If union is 0 (both empty), IoU is 1. If one is empty and other not, IoU is 0.
    # Here target is not empty, pred is empty -> IoU 0 -> Score 0.
    score_zero = compute_map_score(dummy_pred_empty[None, ...], dummy_mask[None, ...])
    assert np.isclose(
        score_zero, 0.0
    ), f"Expected mAP 0.0 for no overlap, got {score_zero}"
    print("mAP calculation verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Initializing DataLoaders ---")
    # Using debug=True to load a small subset for speed
    # load_cached_data=False ensures we test the preprocessing logic
    train_loader, val_loader, _ = get_dataloaders(
        train_csv_path="./metadata/train.csv",
        val_csv_path="./metadata/val.csv",
        batch_size=4,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        load_cached_data=False,
        debug=True,
    )

    # Fetch a batch to verify shapes
    images, masks, ids = next(iter(train_loader))

    # Expected Image Shape: (B, 2, 128, 128) -> 2 channels: Seismic + Depth
    # Expected Mask Shape: (B, 1, 128, 128)
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Mask Shape: {masks.shape}")

    assert images.shape == (4, 2, 128, 128), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (4, 1, 128, 128), f"Unexpected mask shape: {masks.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32"

    print("DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    # in_channels=1 because the model internals add +1 for depth fusion
    model = DeepResUNet(in_channels=1, out_channels=1, depth_fused=True)
    model.to(DEVICE)

    # Verify Forward Pass (Training Mode)
    model.train()
    images = images.to(DEVICE)
    outputs = model(images)

    # DeepResUNet in training returns list of [out128, out64, out32]
    assert isinstance(outputs, list), "Model in train mode should return a list"
    assert len(outputs) == 3, "Model should return 3 outputs for deep supervision"
    assert outputs[0].shape == (4, 1, 128, 128), "Output 128 shape mismatch"
    assert outputs[1].shape == (4, 1, 64, 64), "Output 64 shape mismatch"
    assert outputs[2].shape == (4, 1, 32, 32), "Output 32 shape mismatch"

    print("Model forward pass (training) verification passed.")

    # -------------------------------------------------------------------------
    # 5. Loss Function
    # -------------------------------------------------------------------------
    print("\n--- Initializing Loss Functions ---")
    base_loss = BCEDiceLoss()
    criterion = DeepSupervisionLoss(base_loss, weights=[0.5, 0.3, 0.2])

    masks = masks.to(DEVICE)
    loss = criterion(outputs, masks)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE, epoch=1
    )

    # Check if weights updated (simple check: loss should ideally not be NaN)
    assert not np.isnan(train_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 7. Validation Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Running Validation ---")
    # Note: validate() uses center_crop to return metrics on 101x101 original size
    val_loss, val_map = validate(model, val_loader, criterion, DEVICE)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val mAP: {val_map:.4f}")

    # Save checkpoint
    save_checkpoint(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map": val_map,
        },
        is_best=True,
        checkpoint_dir="./working/demo_checkpoints",
    )

    assert os.path.exists(
        "./working/demo_checkpoints/best_model.pth"
    ), "Checkpoint file not created"
    print("Checkpoint saved successfully.")

    # -------------------------------------------------------------------------
    # 8. Inference / Submission Generation Example
    # -------------------------------------------------------------------------
    print("\n--- Simulating Inference ---")
    model.eval()

    # Take one sample from validation
    sample_img, _, sample_id = next(iter(val_loader))
    sample_img = sample_img[0:1].to(DEVICE)  # (1, 2, 128, 128)

    with torch.no_grad():
        # Model in eval mode returns single tensor
        logits = model(sample_img)
        probs = torch.sigmoid(logits)

        # Crop back to 101x101
        probs_cropped = center_crop(probs, target_h=101, target_w=101)

        # Binarize
        pred_mask = (probs_cropped > 0.5).float().cpu().numpy().squeeze()

    # Encode
    rle_str = rle_encode(pred_mask)
    print(f"Sample ID: {sample_id[0]}")
    print(f"Predicted RLE (first 20 chars): {rle_str[:20]}...")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
