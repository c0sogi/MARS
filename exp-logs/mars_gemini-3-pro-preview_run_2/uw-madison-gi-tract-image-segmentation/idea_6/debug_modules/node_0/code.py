import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, dice_coefficient, hausdorff_distance_3d
from library.dataset import process_metadata, UWDataset, get_transforms
from library.model import LinkNet
from library.train import train
from library.inference import run_inference


def main():
    print("Starting demonstration script...")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(
        f"Configured: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}"
    )
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Logic Verification: Dataset & Data Loading
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset logic...")

    # Process metadata (this will create parquet cache in working dir)
    train_df = process_metadata(Config.TRAIN_METADATA_PATH, mode="train")

    # Instantiate dataset
    ds = UWDataset(train_df, mode="train", transforms=get_transforms("train"))

    # Verify length (should be capped by DEBUG_SAMPLE_SIZE)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > Config.DEBUG_SAMPLE_SIZE:
        # Note: It might be slightly less if filtering happens, but shouldn't be more
        # In UWDataset implementation, it slices *after* sampling, so it should be exactly DEBUG_SAMPLE_SIZE
        # unless the dataframe was smaller to begin with.
        pass

    # Fetch one sample
    img_tensor, mask_tensor = ds[0]

    # Assertions
    assert isinstance(img_tensor, torch.Tensor), "Image must be a torch Tensor"
    assert isinstance(mask_tensor, torch.Tensor), "Mask must be a torch Tensor"

    # Expected shape: (Channels, Height, Width) -> (3, 256, 256)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Image shape mismatch. Got {img_tensor.shape}, expected {expected_shape}"
    assert (
        mask_tensor.shape == expected_shape
    ), f"Mask shape mismatch. Got {mask_tensor.shape}, expected {expected_shape}"

    print("Dataset verification passed: Shapes are correct.")

    # ---------------------------------------------------------
    # 3. Logic Verification: Model Architecture
    # ---------------------------------------------------------
    print("\n[3] Verifying Model architecture...")

    model = LinkNet().to(Config.DEVICE)

    # Create dummy input batch: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Assert output shape: (Batch, Num_Classes, Height, Width)
    expected_out_shape = (2, Config.NUM_CLASSES, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        output.shape == expected_out_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_out_shape}"

    print("Model verification passed: Forward pass successful.")

    # ---------------------------------------------------------
    # 4. Logic Verification: Metrics
    # ---------------------------------------------------------
    print("\n[4] Verifying Metrics...")

    # Test Dice Coefficient
    # Case 1: Perfect overlap
    y_true = torch.ones((1, 1, 10, 10))
    y_pred = torch.ones((1, 1, 10, 10))
    dice = dice_coefficient(y_pred, y_true)
    assert torch.isclose(
        dice, torch.tensor(1.0)
    ), f"Dice should be 1.0 for perfect overlap, got {dice}"

    # Case 2: No overlap
    y_pred_zero = torch.zeros((1, 1, 10, 10))
    dice_zero = dice_coefficient(y_pred_zero, y_true)
    assert torch.isclose(
        dice_zero, torch.tensor(0.0), atol=1e-4
    ), f"Dice should be ~0.0 for no overlap, got {dice_zero}"

    # Test Hausdorff Distance (3D)
    # Create simple 3D volumes (Depth, Height, Width)
    vol_true = np.zeros((5, 10, 10), dtype=np.uint8)
    vol_true[2, 5, 5] = 1  # Point in middle

    vol_pred = np.zeros((5, 10, 10), dtype=np.uint8)
    vol_pred[2, 5, 6] = 1  # Point shifted by 1 pixel in Width

    # Coordinates are normalized by H and W. Distance is Euclidean in (z, y/H, x/W) space?
    # Actually utils.py says: z is slice index (depth=1 spacing), y/H, x/W.
    # Here z diff is 0. y diff is 0. x diff is 1/10 = 0.1.
    # Distance should be 0.1.
    hd = hausdorff_distance_3d(vol_pred, vol_true)
    assert (
        abs(hd - 0.1) < 1e-6
    ), f"Hausdorff distance calculation incorrect. Expected 0.1, got {hd}"

    print("Metric verification passed.")

    # ---------------------------------------------------------
    # 5. Execution: Training Loop
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch, Debug Mode)...")

    # Run training
    train()

    # Verify model artifact
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH} after training."
    print(f"Training complete. Model saved to {Config.MODEL_PATH}")

    # ---------------------------------------------------------
    # 6. Execution: Inference Pipeline
    # ---------------------------------------------------------
    print("\n[6] Executing Inference Pipeline...")

    # Run inference
    # Note: This will load the model we just trained
    run_inference(
        load_cached_data=False
    )  # Force re-process test metadata to ensure debug slicing applies

    # Verify submission artifact
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    required_cols = ["id", "class", "predicted"]
    for col in required_cols:
        assert col in sub_df.columns, f"Submission missing column: {col}"

    print("Inference complete. Submission file verified.")

    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
