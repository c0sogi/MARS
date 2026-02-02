import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coefficient
from library.data import get_dataloaders
from library.model import LightUNet
from library.train import DiceBCELoss, run_training
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_utils():
    print("\n--- Testing Utilities ---")

    # 1. Test Dice Coefficient
    # Case A: Perfect overlap
    pred_perfect = torch.ones((1, 10, 10))
    target_perfect = torch.ones((1, 10, 10))
    dice_perfect = dice_coefficient(pred_perfect, target_perfect)
    assert np.isclose(dice_perfect, 1.0), f"Expected Dice 1.0, got {dice_perfect}"
    print("Dice Coefficient (Perfect Match): Passed")

    # Case B: No overlap
    pred_none = torch.zeros((1, 10, 10))
    target_none = torch.ones((1, 10, 10))
    dice_none = dice_coefficient(pred_none, target_none)
    assert np.isclose(dice_none, 0.0, atol=1e-5), f"Expected Dice 0.0, got {dice_none}"
    print("Dice Coefficient (No Overlap): Passed")

    # 2. Test RLE Encoding
    # Create a simple mask: 3x3, pixels at (0,0), (1,0), (2,0) are 1 (Column 0 is all 1s)
    # Flattened column-major: 1, 1, 1, 0, 0, 0, 0, 0, 0
    # RLE should be: start 1, length 3
    mask = np.zeros((3, 3), dtype=int)
    mask[:, 0] = 1
    rle = rle_encode(mask)
    assert rle == "1 3", f"Expected RLE '1 3', got '{rle}'"

    # Empty mask
    mask_empty = np.zeros((3, 3), dtype=int)
    rle_empty = rle_encode(mask_empty)
    assert rle_empty == "-", f"Expected RLE '-', got '{rle_empty}'"
    print("RLE Encoding: Passed")


def test_data_loading():
    print("\n--- Testing Data Loading ---")

    # Use debug mode to load a small subset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=2, debug=True, debug_sample_size=20
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch
    images, masks, record_ids = next(iter(train_loader))

    # Verify Shapes
    # Config.IN_CHANNELS = 6, Config.IMAGE_SIZE = 256
    expected_img_shape = (4, 6, 256, 256)
    expected_mask_shape = (4, 1, 256, 256)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        masks.shape == expected_mask_shape
    ), f"Mask shape mismatch. Expected {expected_mask_shape}, got {masks.shape}"
    assert len(record_ids) == 4, "Record ID list length mismatch"

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Mask Shape: {masks.shape}")
    print("Data Loading: Passed")

    return images, masks


def test_model_logic(images, masks):
    print("\n--- Testing Model & Loss ---")

    device = torch.device(Config.DEVICE)
    model = LightUNet().to(device)
    criterion = DiceBCELoss()

    images = images.to(device)
    masks = masks.to(device)

    # Forward Pass
    outputs = model(images)

    assert (
        outputs.shape == masks.shape
    ), f"Output shape {outputs.shape} does not match mask shape {masks.shape}"
    assert (
        outputs.min() >= 0 and outputs.max() <= 1
    ), "Model output contains values outside [0, 1]"
    print("Model Forward Pass: Passed")

    # Loss Calculation
    loss = criterion(outputs, masks)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"Calculated Loss: {loss.item():.4f}")
    print("Loss Function: Passed")


def test_training_pipeline():
    print("\n--- Testing Full Training Pipeline ---")

    # Modify Config for fast execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Run Training
    best_model_path = run_training(
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
    )

    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    print(f"Training completed. Model saved to: {best_model_path}")


def test_inference_pipeline():
    print("\n--- Testing Inference Pipeline ---")

    # Run Inference using the model generated in the previous step
    submission_df = generate_submission(
        model_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
        batch_size=Config.BATCH_SIZE,
        debug=True,  # Run on subset of test data
        threshold=0.5,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify DataFrame content
    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "record_id" in submission_df.columns, "Missing 'record_id' column"
    assert "encoded_pixels" in submission_df.columns, "Missing 'encoded_pixels' column"

    print(f"Inference completed. Submission saved to: {Config.SUBMISSION_PATH}")
    print(submission_df.head())


def main():
    # Set seed for reproducibility
    set_seed(42)

    # 1. Test Utilities
    test_utils()

    # 2. Test Data Loading
    images, masks = test_data_loading()

    # 3. Test Model & Logic
    test_model_logic(images, masks)

    # 4. Test Training Pipeline
    test_training_pipeline()

    # 5. Test Inference Pipeline
    test_inference_pipeline()

    print("\nAll tests passed successfully.")


if __name__ == "__main__":
    main()
