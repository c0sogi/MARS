import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    pad_image,
    unpad_image,
    rle_encode,
    rle_decode,
    calculate_iou_map,
)
from library.dataset import get_dataloaders
from library.model import ResNeXtUNet
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.train import Trainer


def run_demo():
    print("=== Starting Salt Segmentation Demo ===\n")

    # 1. Setup and Reproducibility
    set_seed(42)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utility Functions ---")

    # 2.a Padding/Unpadding
    orig_shape = (101, 101)
    dummy_img = np.random.rand(*orig_shape).astype(np.float32)
    padded_img = pad_image(dummy_img)

    assert padded_img.shape == (
        128,
        128,
    ), f"Padding failed. Expected (128, 128), got {padded_img.shape}"

    unpadded_img = unpad_image(padded_img, original_shape=orig_shape)
    assert (
        unpadded_img.shape == orig_shape
    ), f"Unpadding failed. Expected {orig_shape}, got {unpadded_img.shape}"

    # Check if center is preserved (pad uses reflection, so center should be identical)
    assert np.allclose(dummy_img, unpadded_img), "Unpadded image content mismatch!"
    print("Padding/Unpadding logic verified.")

    # 2.b RLE Encoding/Decoding
    # Create a simple mask: 10x10 square of 1s in a 101x101 grid
    mask = np.zeros(orig_shape, dtype=np.uint8)
    mask[10:20, 10:20] = 1

    rle_str = rle_encode(mask)
    decoded_mask = rle_decode(rle_str, shape=orig_shape)

    assert np.array_equal(mask, decoded_mask), "RLE Decode -> Encode mismatch!"
    print("RLE Encoding/Decoding logic verified.")

    # 2.c IoU Metric
    # Perfect match
    iou_score_perfect = calculate_iou_map(mask[None, ...], mask[None, ...])
    assert np.isclose(
        iou_score_perfect, 1.0
    ), f"IoU Perfect match failed. Got {iou_score_perfect}"

    # No match
    empty_mask = np.zeros_like(mask)
    iou_score_mismatch = calculate_iou_map(mask[None, ...], empty_mask[None, ...])
    assert np.isclose(
        iou_score_mismatch, 0.0
    ), f"IoU Mismatch failed. Got {iou_score_mismatch}"
    print("IoU Metric calculation verified.")

    # 3. Verify Data Pipeline
    print("\n--- Verifying Data Pipeline ---")
    # Use debug mode to load a small subset without caching
    batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, batch_size=batch_size, num_workers=2
    )

    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Verify Shapes
    # Config.IN_CHANNELS is 4 (RGB + Depth)
    expected_img_shape = (batch_size, 4, 128, 128)

    assert (
        images.shape == expected_img_shape
    ), f"Image batch shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Check mask shape (B, 1, 128, 128)
    assert (
        masks.ndim == 4 and masks.shape[1] == 1
    ), f"Mask batch shape mismatch. Expected (B, 1, H, W), got {masks.shape}"

    print(f"Data Batch Loaded: Images {images.shape}, Masks {masks.shape}")
    print("Data Pipeline verified.")

    # 4. Verify Model Forward Pass
    print("\n--- Verifying Model Architecture ---")
    model = ResNeXtUNet().to(device)

    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)

    assert (
        logits.shape == masks.shape
    ), f"Model output shape mismatch. Expected {masks.shape}, got {logits.shape}"

    print("Model forward pass successful.")

    # 5. Verify Loss Functions
    print("\n--- Verifying Loss Functions ---")

    # BCE Dice Loss
    criterion1 = BCEDiceLoss()
    loss1 = criterion1(logits, masks)
    assert not torch.isnan(loss1), "BCE Dice Loss returned NaN"
    assert loss1.item() > 0, "BCE Dice Loss should be positive"
    print(f"BCE Dice Loss: {loss1.item():.4f}")

    # Lovasz Hinge Loss
    criterion2 = LovaszHingeLoss()
    loss2 = criterion2(logits, masks)
    assert not torch.isnan(loss2), "Lovasz Hinge Loss returned NaN"
    print(f"Lovasz Hinge Loss: {loss2.item():.4f}")

    # 6. Verify Training Loop (Trainer)
    print("\n--- Verifying Training Loop ---")

    # Configure Trainer for a very short run: 20 samples, 1 epoch each stage
    trainer = Trainer(debug=True, max_samples=20, epochs_stage1=1, epochs_stage2=1)

    # Run training
    best_map = trainer.run()

    print(f"Training finished with Best mAP: {best_map:.4f}")

    # Check if checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"Checkpoint verified at: {checkpoint_path}")

    # 7. Verify Inference / Submission Generation
    print("\n--- Verifying Inference & Submission Format ---")

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get a test batch
    test_images, test_ids = next(iter(test_loader))
    test_images = test_images.to(device)

    with torch.no_grad():
        test_logits = model(test_images)
        test_preds = torch.sigmoid(test_logits)

    # Process first prediction
    pred_mask = test_preds[0].squeeze().cpu().numpy()  # (128, 128)

    # Unpad to original size
    pred_mask_orig = unpad_image(pred_mask)  # (101, 101)

    # Binarize
    binary_mask = (pred_mask_orig > 0.5).astype(np.uint8)

    # Encode
    rle = rle_encode(binary_mask)

    # Simulate submission row
    sub_row = f"{test_ids[0]},{rle}"
    print(f"Sample Submission Row: {sub_row}")

    assert isinstance(rle, str), "RLE should be a string"
    print("Inference and RLE generation verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
