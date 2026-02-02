import os
import numpy as np
import torch
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import (
    SEED,
    DEVICE,
    WORKING_DIR,
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    IN_CHANNELS,
)
from library.utils import (
    seed_everything,
    rle_encode,
    rle_decode,
    keep_largest_component_3d,
)
from library.metrics import compute_dice_coefficient, compute_hausdorff_score
from library.dataset import get_loaders
from library.model import UnetPlusPlus
from library.losses import BCETverskyLoss
from library.train import run_training
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_utilities():
    print("\n=== Testing Utilities ===")

    # 1. Test RLE Encoding/Decoding
    print("Testing RLE Encoding/Decoding...")
    shape = (100, 100)
    mask = np.zeros(shape, dtype=np.uint8)
    # Create a square mask
    mask[20:40, 20:40] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape)

    assert encoded != "", "RLE string should not be empty"
    assert np.array_equal(mask, decoded), "Decoded mask does not match original"
    print("RLE tests passed.")

    # 2. Test 3D Connected Component Analysis (CCA)
    print("Testing 3D CCA (Keep Largest Component)...")
    # Create a 3D volume (Depth, Height, Width)
    vol = np.zeros((5, 20, 20), dtype=np.uint8)

    # Object 1 (Large): 3x3x3 block
    vol[1:4, 5:8, 5:8] = 1
    # Object 2 (Small): 1x1x1 pixel (noise)
    vol[1, 15, 15] = 1

    processed_vol = keep_largest_component_3d(vol)

    # Check that the small object is removed
    assert processed_vol[1, 15, 15] == 0, "Small component should be removed"
    # Check that the large object remains
    assert processed_vol[2, 6, 6] == 1, "Large component should remain"
    assert (
        np.sum(processed_vol) == 27
    ), f"Expected 27 pixels, got {np.sum(processed_vol)}"
    print("3D CCA tests passed.")


def test_metrics():
    print("\n=== Testing Metrics ===")

    # Create dummy masks
    y_true = np.zeros((10, 100, 100), dtype=np.uint8)
    y_true[:, 20:60, 20:60] = 1

    y_pred_perfect = y_true.copy()
    y_pred_empty = np.zeros_like(y_true)

    # 1. Dice Coefficient
    dice_perfect = compute_dice_coefficient(y_true, y_pred_perfect)
    dice_empty = compute_dice_coefficient(y_true, y_pred_empty)

    assert np.isclose(
        dice_perfect, 1.0
    ), f"Perfect Dice should be 1.0, got {dice_perfect}"
    assert np.isclose(dice_empty, 0.0), f"Empty Dice should be 0.0, got {dice_empty}"
    print("Dice metric tests passed.")

    # 2. Hausdorff Score
    # Perfect match -> Distance 0 -> Score 1.0
    hd_perfect = compute_hausdorff_score(y_true, y_pred_perfect)
    assert np.isclose(
        hd_perfect, 1.0
    ), f"Perfect HD Score should be 1.0, got {hd_perfect}"

    # One empty -> Score 0.0
    hd_empty = compute_hausdorff_score(y_true, y_pred_empty)
    assert np.isclose(hd_empty, 0.0), f"Empty HD Score should be 0.0, got {hd_empty}"
    print("Hausdorff metric tests passed.")


def test_data_pipeline():
    print("\n=== Testing Data Pipeline ===")

    # Initialize loaders in debug mode (small subset)
    train_loader, val_loader = get_loaders(load_cached_data=True, debug=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Verify shapes
    # Images: (B, C, H, W) -> C=3 for 2.5D (t-1, t, t+1)
    # Masks: (B, C, H, W) -> C=3 for (Large Bowel, Small Bowel, Stomach)
    print(f"Image Batch Shape: {images.shape}")
    print(f"Mask Batch Shape: {masks.shape}")

    assert len(images.shape) == 4, "Images should be 4D tensor"
    assert images.shape[1] == IN_CHANNELS, f"Expected {IN_CHANNELS} input channels"
    assert masks.shape[1] == NUM_CLASSES, f"Expected {NUM_CLASSES} mask channels"
    assert (
        images.shape[2:] == IMAGE_SIZE
    ), f"Image spatial dims mismatch {images.shape[2:]} vs {IMAGE_SIZE}"

    print("Data pipeline tests passed.")
    return train_loader


def test_model_and_loss(train_loader):
    print("\n=== Testing Model and Loss ===")

    # Instantiate model
    model = UnetPlusPlus().to(DEVICE)
    criterion = BCETverskyLoss().to(DEVICE)

    # Get a batch
    images, masks = next(iter(train_loader))
    images = images.to(DEVICE, dtype=torch.float32)
    masks = masks.to(DEVICE, dtype=torch.float32)

    # Forward pass
    outputs = model(images)

    # With Deep Supervision, output is a list of tensors
    assert isinstance(outputs, list), "Model with deep supervision should return a list"
    print(f"Number of deep supervision outputs: {len(outputs)}")

    main_output = outputs[0]
    assert (
        main_output.shape == masks.shape
    ), f"Output shape {main_output.shape} mismatch with mask {masks.shape}"

    # Calculate loss
    loss = criterion(outputs, masks)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Model and Loss tests passed.")


def run_full_training_cycle():
    print("\n=== Running Full Training Cycle (Debug Mode) ===")

    # Run training for 1 epoch using the library function
    # debug=True ensures we use a tiny subset of data
    run_training(debug=True, epochs=1)

    # Verify checkpoint creation
    expected_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), f"Checkpoint not found at {expected_ckpt}"
    print("Training cycle completed and checkpoint verified.")

    return expected_ckpt


def run_inference_cycle(checkpoint_path):
    print("\n=== Running Inference Cycle ===")

    output_dir = os.path.join(WORKING_DIR, "submission_test")
    output_file = "submission.csv"

    # Run inference
    predict_and_submit(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        output_filename=output_file,
    )

    # Verify submission file
    submission_path = os.path.join(output_dir, output_file)
    assert os.path.exists(submission_path), "Submission file not created"

    df = pd.read_csv(submission_path)
    print(f"Submission rows: {len(df)}")
    print("Submission Head:")
    print(df.head())

    required_cols = ["id", "class", "predicted"]
    assert all(
        col in df.columns for col in required_cols
    ), "Missing columns in submission"
    assert len(df) > 0, "Submission file is empty"

    print("Inference cycle tests passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(SEED)

    # 1. Test Utilities
    test_utilities()

    # 2. Test Metrics
    test_metrics()

    # 3. Test Data Pipeline
    loader = test_data_pipeline()

    # 4. Test Model & Loss
    test_model_and_loss(loader)

    # 5. Run Training Loop (1 Epoch, Debug Data)
    ckpt_path = run_full_training_cycle()

    # 6. Run Inference
    run_inference_cycle(ckpt_path)

    print("\nAll demonstrations completed successfully.")
