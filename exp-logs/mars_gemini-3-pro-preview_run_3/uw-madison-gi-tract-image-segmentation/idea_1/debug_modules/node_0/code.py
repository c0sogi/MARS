import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, dice_coef, compute_hausdorff_3d, rle_decode
from library.dataset import (
    process_and_cache_25d_metadata,
    GI_MRI_Dataset,
    get_transforms,
)
from library.model import FPN
from library.train import train_model
from library.inference import run_inference

if __name__ == "__main__":
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    # We use debug=True to drastically reduce runtime (fewer epochs, subset of data)
    Config.setup(debug=True, training=True)
    set_seed(Config.SEED)
    print(f"Configuration loaded. Device: {Config.DEVICE}, Debug Mode: {Config.DEBUG}")

    # 2. Data Processing (2.5D Metadata)
    print("\n--- Testing Metadata Processing ---")
    # This function adds 'prev_image_path' and 'next_image_path' columns
    train_df, val_df, test_df = process_and_cache_25d_metadata(load_cached_data=False)

    # Validation
    assert "prev_image_path" in train_df.columns, "Metadata missing 'prev_image_path'"
    assert "next_image_path" in train_df.columns, "Metadata missing 'next_image_path'"
    assert len(train_df) > 0, "Training dataframe is empty"
    print(f"Metadata processed successfully. Train rows: {len(train_df)}")

    # 3. Dataset and Transforms
    print("\n--- Testing Dataset and Transforms ---")
    # Create a small subset for testing dataset class logic
    subset_df = train_df.head(10).copy()
    dataset = GI_MRI_Dataset(
        df=subset_df, transforms=get_transforms(mode="train"), mode="train"
    )

    # Fetch one sample
    img_tensor, mask_tensor = dataset[0]

    # Validation
    # Image shape should be (Channels, Height, Width). Channels=3 for 2.5D input.
    expected_img_shape = (Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    # Mask shape should be (Classes, Height, Width).
    expected_mask_shape = (Config.NUM_CLASSES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    print(f"Image Tensor Shape: {img_tensor.shape}")
    print(f"Mask Tensor Shape: {mask_tensor.shape}")

    assert (
        img_tensor.shape == expected_img_shape
    ), f"Expected image shape {expected_img_shape}, got {img_tensor.shape}"
    assert (
        mask_tensor.shape == expected_mask_shape
    ), f"Expected mask shape {expected_mask_shape}, got {mask_tensor.shape}"
    assert img_tensor.dtype == torch.float32, "Image tensor should be float32"
    assert mask_tensor.dtype == torch.float32, "Mask tensor should be float32"

    # Check normalization (approximate 0-1 range)
    assert (
        img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    ), "Image tensor not normalized to [0, 1]"

    # 4. Model Architecture
    print("\n--- Testing Model Architecture ---")
    model = FPN(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # Speed up init
        num_classes=Config.NUM_CLASSES,
    ).to(Config.DEVICE)

    # Dummy forward pass
    # Add batch dimension: (1, 3, 320, 320)
    input_batch = img_tensor.unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        output = model(input_batch)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        1,
        Config.NUM_CLASSES,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Model output shape mismatch"
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output (sigmoid) should be in [0, 1]"

    # 5. Metrics Verification
    print("\n--- Testing Metrics ---")
    # Create synthetic perfect match data
    y_true = torch.zeros((1, 3, 32, 32))
    y_true[:, :, 10:20, 10:20] = 1.0
    y_pred = y_true.clone()

    # Test Dice
    score = dice_coef(y_pred, y_true)
    print(f"Dice Score (Perfect Match): {score.item():.4f}")
    assert torch.isclose(
        score, torch.tensor(1.0)
    ), "Dice score for perfect match should be 1.0"

    # Test Hausdorff 3D
    # Create dummy volumes (Depth, Height, Width)
    vol_true = np.zeros((5, 100, 100), dtype=np.uint8)
    vol_true[2, 50, 50] = 1
    vol_pred = np.zeros((5, 100, 100), dtype=np.uint8)
    vol_pred[2, 50, 51] = 1  # 1 pixel offset

    hd_dist = compute_hausdorff_3d(
        vol_pred, vol_true, spacing_h=1.0, spacing_w=1.0, slice_depth=1.0
    )
    print(f"Hausdorff Distance (1px offset): {hd_dist:.4f}")
    assert (
        hd_dist > 0
    ), "Hausdorff distance should be positive for non-overlapping/offset pixels"

    # 6. Training Loop (Debug Mode)
    print("\n--- Running Training Loop (Debug Mode) ---")
    # train_model(debug=True) uses a small subset of data and runs for fewer epochs (defined in Config.setup)
    best_dice = train_model(debug=True)

    print(f"Training finished. Best Validation Dice: {best_dice:.4f}")
    assert 0.0 <= best_dice <= 1.0, "Best dice score out of range"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "best_model.pth")
    ), "Model checkpoint not saved"

    # 7. Inference Pipeline
    print("\n--- Running Inference Pipeline (Debug Mode) ---")
    # run_inference(debug=True) predicts on a subset of test data
    submission_df = run_inference(debug=True)

    print("Submission DataFrame Head:")
    print(submission_df.head())

    # Validation of submission
    required_cols = ["id", "class", "predicted"]
    for col in required_cols:
        assert col in submission_df.columns, f"Submission missing column: {col}"

    assert len(submission_df) > 0, "Submission dataframe is empty"

    # Check RLE format (should be string, space delimited integers)
    sample_rle = submission_df.iloc[0]["predicted"]
    if sample_rle != "":
        parts = sample_rle.split()
        assert all(p.isdigit() for p in parts), "RLE contains non-digit characters"
        assert len(parts) % 2 == 0, "RLE must have pairs of values"

    print("\n=== Demonstration Complete: All Tests Passed ===")
