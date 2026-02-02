import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, dice_coef, hausdorff_3d
from library.dataset import process_metadata, UWGI25DDataset, get_dataloaders
from library.ghost_model import GhostUNet
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration for Speed
    # Enable debug mode to use a small subset of data and fewer epochs
    print("\n[1] Configuring environment...")
    Config.set_debug_mode(debug=True)

    # Ensure clean working directory for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading and Processing...")

    # Process metadata (this creates the cache)
    train_df = process_metadata(
        Config.TRAIN_METADATA_PATH, "train_demo", load_cached_data=False
    )

    # Verify DataFrame structure
    required_cols = ["id", "case", "day", "slice", "abs_path", "prev_path", "next_path"]
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in processed metadata"

    print(f"Processed Train DataFrame shape: {train_df.shape}")

    # Instantiate Dataset
    # We take a small sample for verification
    sample_df = train_df.head(10).copy()
    dataset = UWGI25DDataset(sample_df, transforms=None, mode="train")

    # Fetch one item
    img_stack, mask, img_id = dataset[0]

    # Verify shapes
    # Image: (Channels, Height, Width) -> (3, 256, 256)
    # Mask: (Classes, Height, Width) -> (3, 256, 256)
    print(f"Sample Image Shape: {img_stack.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    assert img_stack.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Incorrect image stack shape"
    assert mask.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Incorrect mask shape"
    assert isinstance(img_stack, torch.Tensor), "Image should be a tensor"
    assert isinstance(mask, torch.Tensor), "Mask should be a tensor"

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GhostUNet(
        in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES
    ).to(device)

    # Create a dummy batch
    dummy_input = torch.randn(2, 3, 256, 256).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 3, 256, 256), "Model output shape mismatch"

    # Clean up
    del model, dummy_input, output
    torch.cuda.empty_cache()

    # 4. Run Training Loop
    print("\n[4] Running Training Loop (Debug Mode)...")
    # This calls the library function which handles the full training lifecycle
    run_training()

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH}"
    print("Training completed successfully.")

    # 5. Run Inference Loop
    print("\n[5] Running Inference Loop...")
    # This calls the library function which handles prediction and submission generation
    run_inference()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"
    print("Inference completed successfully.")

    # 6. Submission Verification
    print("\n[6] Verifying Submission File...")
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["id", "class", "predicted"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Found: {sub_df.columns}"

    # Check content
    # In debug mode with very little training, predictions might be empty or sparse,
    # but the structure must be valid.
    print(f"Submission rows: {len(sub_df)}")
    print("Sample rows:")
    print(sub_df.head())

    # Ensure 'predicted' column contains strings (even empty ones)
    assert (
        sub_df["predicted"].dtype == object
    ), "Predicted column should be object/string type"

    # 7. Metric Unit Tests
    print("\n[7] Verifying Metrics Logic...")

    # Test Dice Coefficient
    # Perfect match
    y_true = np.ones((10, 10))
    y_pred = np.ones((10, 10))
    d = dice_coef(y_true, y_pred)
    assert np.isclose(d, 1.0), f"Dice should be 1.0 for perfect match, got {d}"

    # No overlap
    y_pred_zero = np.zeros((10, 10))
    d_zero = dice_coef(y_true, y_pred_zero)
    assert np.isclose(
        d_zero, 0.0, atol=1e-4
    ), f"Dice should be ~0.0 for no overlap, got {d_zero}"

    # Test 3D Hausdorff
    # Create simple 3D volumes (Depth, Height, Width)
    vol_true = np.zeros((5, 10, 10))
    vol_true[2, 5, 5] = 1  # One pixel in the middle

    vol_pred_perfect = vol_true.copy()

    # Perfect match -> Distance 0
    h_perfect = hausdorff_3d(vol_true, vol_pred_perfect)
    assert (
        h_perfect == 0.0
    ), f"Hausdorff should be 0.0 for perfect match, got {h_perfect}"

    # Offset by 1 pixel in H dimension
    vol_pred_offset = np.zeros((5, 10, 10))
    vol_pred_offset[2, 6, 5] = 1

    # Distance calculation:
    # Coordinates are normalized by H and W. Z is not normalized (step 1).
    # Point 1: (2, 5/10, 5/10) = (2, 0.5, 0.5)
    # Point 2: (2, 6/10, 5/10) = (2, 0.6, 0.5)
    # Euclidean distance = sqrt(0 + 0.1^2 + 0) = 0.1
    h_offset = hausdorff_3d(vol_true, vol_pred_offset)
    assert np.isclose(
        h_offset, 0.1
    ), f"Hausdorff should be 0.1 for 1 pixel offset in 10x10 image, got {h_offset}"

    print("Metrics verification passed.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
