import os
import torch
import numpy as np
import pandas as pd
import time
import shutil

# Import from the provided library
from library.config import Config, setup_system
from library.dataset import UWDataset
from library.model import MobileNetV2UNet
from library.loss import BCEDiceLoss
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import dice_coef, hausdorff_3d, rle_encode, rle_decode


def main():
    print("=== Starting Demonstration and Verification Script ===")

    # 1. System Setup
    print("\n[Step 1] Setting up system environment...")
    setup_system(seed=42)

    # Ensure clean state for working directory in this run
    if os.path.exists(Config.CACHE_DIR):
        print(f"Cleaning cache directory: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Dataset Verification
    print("\n[Step 2] Verifying Dataset Loading (Debug Mode)...")
    # Initialize dataset in debug mode (loads small subset)
    train_ds = UWDataset(mode="train", debug=True, load_cached_data=False)

    print(f"Dataset length (debug): {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset should not be empty."

    # Fetch a sample
    image, mask = train_ds[0]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Verify Shapes
    # Image: (3, 256, 256) -> [Prev, Curr, Next] channels, Height, Width
    assert image.shape == (
        3,
        256,
        256,
    ), f"Expected image shape (3, 256, 256), got {image.shape}"
    # Mask: (3, 256, 256) -> [Large Bowel, Small Bowel, Stomach], Height, Width
    assert mask.shape == (
        3,
        256,
        256,
    ), f"Expected mask shape (3, 256, 256), got {mask.shape}"

    # Verify Data Types and Ranges
    assert image.dtype == torch.float32, "Image should be float32"
    assert mask.dtype == torch.float32, "Mask should be float32"
    assert (
        image.min() >= 0.0 and image.max() <= 1.0
    ), "Image values should be normalized to [0, 1]"
    assert (
        mask.min() >= 0.0 and mask.max() <= 1.0
    ), "Mask values should be binary (0 or 1)"

    print("Dataset verification passed.")

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture...")
    model = MobileNetV2UNet(
        pretrained=False
    )  # No need to download weights for shape check
    model.eval()

    # Create a dummy batch (Batch Size = 2)
    dummy_input = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Verify Output Shape: (Batch, Num_Classes, H, W)
    assert output.shape == (
        2,
        3,
        256,
        256,
    ), f"Expected output shape (2, 3, 256, 256), got {output.shape}"

    # Verify Output Range (Sigmoid applied manually for check)
    output_prob = torch.sigmoid(output)
    assert (
        output_prob.min() >= 0.0 and output_prob.max() <= 1.0
    ), "Model output (after sigmoid) should be probabilities [0, 1]"

    print("Model architecture verification passed.")

    # 4. Loss Function and Metrics Verification
    print("\n[Step 4] Verifying Loss and Metrics...")
    criterion = BCEDiceLoss()

    # Create synthetic ground truth and prediction
    # y_true: (2, 3, 256, 256)
    y_true = torch.zeros((2, 3, 256, 256))
    y_true[:, :, 100:150, 100:150] = 1.0

    # y_pred (Logits):
    # Perfect match: Large positive value where true is 1, large negative where true is 0
    y_pred_perfect_logits = torch.ones_like(y_true) * -10.0
    y_pred_perfect_logits[y_true == 1] = 10.0

    # Bad match: Opposite
    y_pred_bad_logits = torch.ones_like(y_true) * 10.0
    y_pred_bad_logits[y_true == 1] = -10.0

    # Test Loss
    loss_perfect = criterion(y_pred_perfect_logits, y_true)
    loss_bad = criterion(y_pred_bad_logits, y_true)

    print(f"Loss (Perfect Match): {loss_perfect.item():.6f}")
    print(f"Loss (No Overlap): {loss_bad.item():.6f}")

    # Ideally perfect match loss should be close to 0 (BCELoss(0) + DiceLoss(0))
    # Due to smoothing in Dice, it might not be exactly 0 but very small.
    assert (
        loss_perfect < loss_bad
    ), "Perfect match should have lower loss than bad prediction."

    # Test Dice Coefficient (Requires Probabilities)
    y_pred_perfect_prob = torch.sigmoid(y_pred_perfect_logits)
    y_pred_bad_prob = torch.sigmoid(y_pred_bad_logits)

    dice_perfect = dice_coef(y_true, y_pred_perfect_prob)
    dice_bad = dice_coef(y_true, y_pred_bad_prob)

    print(f"Dice (Perfect Match): {dice_perfect:.6f}")
    print(f"Dice (No Overlap): {dice_bad:.6f}")

    assert dice_perfect > 0.99, "Dice score for perfect match should be ~1.0"
    assert dice_bad < 0.01, "Dice score for no overlap should be ~0.0"

    # Test Hausdorff 3D
    # Construct 3D volumes (Depth=2, H=256, W=256)
    vol_true = y_true[0, 0].numpy()[
        None, :, :
    ]  # Take first sample, first class -> (1, 256, 256)
    vol_pred = (y_pred_perfect_prob[0, 0] > 0.5).float().numpy()[None, :, :]

    # Perfect match Hausdorff should be 0
    hd_perfect = hausdorff_3d(vol_true, vol_pred)
    print(f"Hausdorff (Perfect Match): {hd_perfect:.6f}")
    assert hd_perfect == 0.0, "Hausdorff distance for perfect match should be 0.0"

    print("Loss and metrics verification passed.")

    # 5. Training Loop Verification
    print("\n[Step 5] Verifying Training Loop (1 Epoch)...")

    # Initialize Trainer with debug=True
    trainer = Trainer(debug=True, load_cached_data=False)

    # Run training for 1 epoch
    # This verifies the integration of Dataset, DataLoader, Model, Loss, and Optimizer
    trainer.fit(epochs=1)

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print(f"Training loop completed. Model saved to {Config.MODEL_SAVE_PATH}")

    # 6. Inference and Submission Verification
    print("\n[Step 6] Verifying Inference and Submission Generation...")

    # Generate submission using the model trained in Step 5
    # Using debug=True to run on a subset of test data
    submission_df = generate_submission(debug=True, load_cached_data=False)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify DataFrame structure
    expected_cols = ["id", "class", "predicted"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(submission_df.columns)}"

    # Verify content
    assert len(submission_df) > 0, "Submission dataframe is empty"

    # Check RLE format (basic check)
    # The 'predicted' column should contain strings (can be empty string for no mask)
    assert (
        submission_df["predicted"].dtype == object
    ), "Predicted column should be object (string)"

    print("Inference verification passed.")
    print("\n=== All Demonstrations and Verifications Completed Successfully ===")


if __name__ == "__main__":
    main()
