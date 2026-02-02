import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    rle_encode,
    rle_decode,
    compute_dice_score,
    compute_hausdorff_3d,
    keep_largest_component_3d,
)
from library.dataset import UWGIDataset, get_transforms
from library.model import UNet25D
from library.losses import BCEDiceLoss
from library.train import run_training
from library.inference import run_inference


def verify_utils():
    print("=== Verifying Utils ===")

    # 1. Test RLE Encoding/Decoding
    # Create a 100x100 mask with a 10x10 square
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 20:30] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, (100, 100))

    assert np.array_equal(mask, decoded), "RLE Decode does not match original mask"
    print("  [Pass] RLE Encode/Decode")

    # 2. Test Metrics
    # Perfect overlap
    y_true = np.zeros((5, 100, 100), dtype=np.uint8)
    y_true[2, 20:30, 20:30] = 1
    y_pred = y_true.copy()

    dice = compute_dice_score(y_pred, y_true)
    hd = compute_hausdorff_3d(y_pred, y_true)

    assert np.isclose(
        dice, 1.0
    ), f"Dice score for perfect match should be 1.0, got {dice}"
    assert np.isclose(
        hd, 0.0
    ), f"Hausdorff distance for perfect match should be 0.0, got {hd}"
    print("  [Pass] Metrics (Dice & Hausdorff)")

    # 3. Test 3D Post-processing (Largest Component)
    # Create volume with one large component (100 pixels) and one small noise (1 pixel)
    vol = np.zeros((5, 100, 100), dtype=np.uint8)
    vol[2, 20:30, 20:30] = 1  # Size 100
    vol[0, 5, 5] = 1  # Size 1

    processed_vol = keep_largest_component_3d(vol)

    assert (
        processed_vol.sum() == 100
    ), f"Expected 100 pixels after cleaning, got {processed_vol.sum()}"
    assert processed_vol[0, 5, 5] == 0, "Noise pixel was not removed"
    print("  [Pass] 3D Largest Component Retention")


def verify_dataset_and_model():
    print("\n=== Verifying Dataset and Model ===")

    # Load metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_CSV}")

    df = pd.read_csv(Config.TRAIN_CSV, keep_default_na=False)

    # Use a tiny subset for verification
    subset_df = df.iloc[:10].reset_index(drop=True)

    # Instantiate Dataset
    dataset = UWGIDataset(subset_df, transforms=get_transforms("train"), mode="train")

    # Fetch one sample
    img, mask, sample_id = dataset[0]

    # Verify Shapes
    # Image: (Channels, Height, Width) -> (3, 320, 320)
    # Mask: (Channels, Height, Width) -> (3, 320, 320)
    print(f"  Sample ID: {sample_id}")
    print(f"  Image Shape: {img.shape}")
    print(f"  Mask Shape: {mask.shape}")

    assert img.shape == (3, 320, 320), f"Unexpected image shape: {img.shape}"
    assert mask.shape == (3, 320, 320), f"Unexpected mask shape: {mask.shape}"
    assert img.dtype == torch.float32, "Image should be float32"

    # Instantiate Model
    device = torch.device(Config.DEVICE)
    model = UNet25D(classes=Config.NUM_CLASSES).to(device)

    # Forward Pass
    input_tensor = (
        torch.from_numpy(img).unsqueeze(0).to(device)
    )  # Add batch dim -> (1, 3, 320, 320)
    with torch.no_grad():
        output = model(input_tensor)

    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (1, 3, 320, 320), f"Unexpected output shape: {output.shape}"

    # Loss Calculation
    criterion = BCEDiceLoss()
    target_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    loss = criterion(output, target_tensor)

    print(f"  Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("  [Pass] Dataset loading, Model forward pass, and Loss calculation")


def verify_training_pipeline():
    print("\n=== Verifying Training Pipeline ===")

    # Run training in debug mode
    # debug=True forces: subset of data (200 train, 100 val) and epochs=2
    # This ensures it runs quickly.
    print("  Starting training run (Debug Mode)...")
    run_training(epochs=2, batch_size=4, debug=True, patience=2)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"  [Pass] Checkpoint created at {checkpoint_path}")
    else:
        raise FileNotFoundError("Checkpoint was not created during training.")


def verify_inference_pipeline():
    print("\n=== Verifying Inference Pipeline ===")

    # To make inference fast for demonstration, we create a small subset of test metadata
    # and temporarily point Config.TEST_CSV to it.

    original_test_csv_path = Config.TEST_CSV
    temp_test_csv_path = os.path.join(Config.WORKING_DIR, "temp_test_subset.csv")

    try:
        # Load full test metadata
        full_test_df = pd.read_csv(original_test_csv_path, keep_default_na=False)

        # Take a small subset (e.g., slices from the first 2 cases)
        # We need enough slices to form a volume for 3D processing check
        cases = full_test_df["case"].unique()[:2]
        subset_test_df = full_test_df[full_test_df["case"].isin(cases)].copy()

        print(f"  Creating temporary test subset with {len(subset_test_df)} slices...")
        subset_test_df.to_csv(temp_test_csv_path, index=False)

        # Monkey-patch Config to use the temp file
        Config.TEST_CSV = temp_test_csv_path

        # Run Inference
        print("  Starting inference run...")
        run_inference()

        # Verify Submission
        submission_path = Config.SUBMISSION_FILE
        if os.path.exists(submission_path):
            sub_df = pd.read_csv(submission_path)
            print(f"  Submission generated with {len(sub_df)} rows.")

            # Check columns
            expected_cols = ["id", "class", "predicted"]
            if list(sub_df.columns) == expected_cols:
                print("  [Pass] Submission format correct.")
            else:
                raise ValueError(
                    f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
                )

            # Check if predictions are RLE strings (or empty)
            # Just check type of first prediction
            if len(sub_df) > 0:
                pred_val = sub_df.iloc[0]["predicted"]
                # It can be a string "1 10..." or NaN/float if empty and loaded weirdly,
                # but pandas usually loads empty strings as NaN unless keep_default_na=False.
                # The inference script saves it, so we just check existence.
                pass
        else:
            raise FileNotFoundError("Submission file not found.")

    finally:
        # Restore Config (good practice, though script ends here)
        Config.TEST_CSV = original_test_csv_path
        # Cleanup temp file
        if os.path.exists(temp_test_csv_path):
            os.remove(temp_test_csv_path)


if __name__ == "__main__":
    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    try:
        verify_utils()
        verify_dataset_and_model()
        verify_training_pipeline()
        verify_inference_pipeline()
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
