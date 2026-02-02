import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2
from functools import partial

# Monkeypatch tqdm to suppress progress bars as per requirements
# This must be done before importing modules that use tqdm
import tqdm

_original_tqdm = tqdm.tqdm


def silent_tqdm(*args, **kwargs):
    kwargs["disable"] = True
    return _original_tqdm(*args, **kwargs)


sys.modules["tqdm"].tqdm = silent_tqdm

# Import library modules
from library.utils import set_seed, rle_encode, calc_map_score
from library.losses import CombinedLoss, StableBCELoss, LovaszHingeLoss
from library.model import ResNet34WideLinkNet
from library.dataset import get_loaders, pad_image
from library.engine import train_one_epoch, evaluate, predict_proba


def main():
    print("Initializing demonstration...")

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # 2. Verify Utils
    print("Verifying Utils...")
    # Test RLE Encode with a simple 3x3 pattern
    # Pattern:
    # 0 1 0
    # 1 1 1
    # 0 0 0
    # Flattened (Fortran/Column-major): 0, 1, 0, 1, 1, 0, 0, 1, 0
    # Indices (1-based):                1, 2, 3, 4, 5, 6, 7, 8, 9
    # Runs:
    # Start 2, Length 1
    # Start 4, Length 2
    # Start 8, Length 1
    # Expected RLE string: "2 1 4 2 8 1"
    dummy_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    rle_str = rle_encode(dummy_mask)
    assert (
        rle_str == "2 1 4 2 8 1"
    ), f"RLE Encode failed. Expected '2 1 4 2 8 1', got '{rle_str}'"

    # Test mAP Score
    score_perfect = calc_map_score(dummy_mask, dummy_mask)
    assert np.isclose(score_perfect, 1.0), "mAP should be 1.0 for perfect match"

    score_mismatch = calc_map_score(dummy_mask, np.zeros_like(dummy_mask))
    assert np.isclose(score_mismatch, 0.0), "mAP should be 0.0 for complete mismatch"
    print("Utils verification passed.")

    # 3. Verify Dataset Loading
    print("Verifying Dataset...")
    # Use debug=True to load a small subset (100 train, 50 val, 50 test)
    # load_cached_data=False ensures we test the preprocessing logic
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=4, debug=True, load_cached_data=False
    )

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Verify shapes
    # Images: (B, C, H, W) -> (4, 1, 128, 128)
    assert images.dim() == 4, f"Image dim mismatch: {images.shape}"
    assert images.size(1) == 1, "Images should be 1-channel (grayscale)"
    assert (
        images.size(2) == 128 and images.size(3) == 128
    ), "Images should be padded to 128x128"

    # Masks: (B, 1, 128, 128)
    assert masks.dim() == 4, f"Mask dim mismatch: {masks.shape}"

    # Depths: (B, 1)
    assert depths.dim() == 2, f"Depth dim mismatch: {depths.shape}"
    assert depths.size(1) == 1, "Depth should have shape (B, 1)"

    print("Dataset verification passed.")

    # 4. Verify Model
    print("Verifying Model...")
    model = ResNet34WideLinkNet().to(device)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Forward pass
    logits = model(images, depths)
    assert logits.shape == (
        4,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {logits.shape}"
    print("Model verification passed.")

    # 5. Verify Loss
    print("Verifying Loss...")
    criterion = CombinedLoss().to(device)
    loss = criterion(logits, masks)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Check backward pass
    loss.backward()
    print("Loss verification passed.")

    # 6. Verify Training Engine
    print("Verifying Training Engine...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for one epoch on the debug subset
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)
    print(f"Train Epoch Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # 7. Verify Evaluation Engine
    print("Verifying Evaluation Engine...")
    val_loss, val_map = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")
    assert val_loss > 0, "Validation loss should be positive"
    assert 0.0 <= val_map <= 1.0, "mAP should be between 0 and 1"

    # 8. Verify Prediction and Post-processing
    print("Verifying Prediction & Post-processing...")
    preds_dict = predict_proba(model, test_loader, device)

    assert len(preds_dict) > 0, "No predictions generated"

    # Take the first prediction
    sample_id = list(preds_dict.keys())[0]
    sample_pred = preds_dict[sample_id]  # Shape (128, 128)

    assert sample_pred.shape == (128, 128), "Raw prediction should be 128x128"

    # Post-processing: Crop back to 101x101
    # Logic matches dataset.py:pad_image
    target_size = 128
    orig_size = 101
    pad_h = target_size - orig_size
    pad_w = target_size - orig_size
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    cropped_pred = sample_pred[
        pad_top : pad_top + orig_size, pad_left : pad_left + orig_size
    ]
    assert cropped_pred.shape == (
        101,
        101,
    ), f"Cropped prediction should be 101x101, got {cropped_pred.shape}"

    # Binarize and Encode
    binary_mask = (cropped_pred > 0.5).astype(np.uint8)
    rle_result = rle_encode(binary_mask)

    print(f"Sample ID: {sample_id}")
    print(f"RLE Result (truncated): {rle_result[:50]}...")

    print("All verifications passed successfully.")


if __name__ == "__main__":
    main()
