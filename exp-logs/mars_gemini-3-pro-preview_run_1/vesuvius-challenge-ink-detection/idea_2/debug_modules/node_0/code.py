import os
import numpy as np
import pandas as pd
import torch
import shutil

# Import provided library modules
from library.config import Config
from library.utils import rle_encode, calculate_fbeta
from library.dataset import InkDataset
from library.model import InkUNet
from library.train import run_training


def create_mini_test_metadata():
    """Creates a small subset of test metadata for rapid inference testing."""
    original_test_path = Config.TEST_METADATA_PATH
    if os.path.exists(original_test_path):
        df = pd.read_csv(original_test_path)
        # Take only 2 patches to speed up the inference step in run_training
        mini_df = df.head(2).copy()

        mini_path = os.path.join(Config.WORKING_DIR, "test_mini.csv")
        mini_df.to_csv(mini_path, index=False)

        # Override Config to point to this mini file
        Config.TEST_METADATA_PATH = mini_path
        print(f"Created mini test metadata at {mini_path} with {len(mini_df)} samples.")


def main():
    print(">>> Starting Ink Detection Demo Script")

    # 1. Setup & Configuration Overrides
    # We modify Config attributes directly to ensure the demo runs quickly within limits.
    Config.setup()
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo stability
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "demo_checkpoints")

    # Create working directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Create a mini test set to ensure generate_submission finishes in seconds
    create_mini_test_metadata()

    # 2. Verify Utility Functions
    print("\n>>> Verifying Utils...")

    # Test RLE Encoding
    # Mask: 0 1 1 0 0 1 -> Indices (1-based): 2, 3, 6
    # RLE: Start 2 Len 2, Start 6 Len 1 -> "2 2 6 1"
    dummy_mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    rle_result = rle_encode(dummy_mask)
    assert rle_result == "2 2 6 1", f"RLE verification failed. Got: {rle_result}"

    # Test F-Beta Score
    # Pred: 1 1 0 0, True: 1 0 1 0 -> TP=1, FP=1, FN=1
    # Beta=0.5 -> Score = 0.5
    y_pred = np.array([1, 1, 0, 0])
    y_true = np.array([1, 0, 1, 0])
    fbeta = calculate_fbeta(y_pred, y_true, beta=0.5)
    assert abs(fbeta - 0.5) < 1e-6, f"F-beta verification failed. Got: {fbeta}"
    print("Utils verified successfully.")

    # 3. Verify Dataset Loading
    print("\n>>> Verifying Dataset...")
    # Load only 4 samples to be fast
    train_ds = InkDataset(mode="train", limit=4, load_cached_data=False)
    assert len(train_ds) == 4, "Dataset limit not respected."

    # Fetch one sample
    vol_tensor, label_tensor = train_ds[0]

    # Check shapes
    # Volume: (Z_DIM, PATCH_SIZE, PATCH_SIZE)
    # Label: (1, PATCH_SIZE, PATCH_SIZE)
    expected_vol_shape = (Config.Z_DIM, Config.PATCH_SIZE, Config.PATCH_SIZE)
    expected_label_shape = (1, Config.PATCH_SIZE, Config.PATCH_SIZE)

    assert (
        vol_tensor.shape == expected_vol_shape
    ), f"Volume shape mismatch. Expected {expected_vol_shape}, got {vol_tensor.shape}"
    assert (
        label_tensor.shape == expected_label_shape
    ), f"Label shape mismatch. Expected {expected_label_shape}, got {label_tensor.shape}"

    print("Dataset verified successfully.")

    # 4. Verify Model Architecture
    print("\n>>> Verifying Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InkUNet(z_dim=Config.Z_DIM).to(device)

    # Create a dummy input with reduced spatial dimensions (64x64) to test forward pass speed
    # The U-Net requires input to be divisible by 32 (2^5 downsampling)
    dummy_input = torch.randn(2, Config.Z_DIM, 64, 64).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Output should be (B, 1, H, W)
    expected_out_shape = (2, 1, 64, 64)
    assert (
        output.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {output.shape}"

    print("Model verified successfully.")

    # 5. Integration Test: Full Training Pipeline
    print("\n>>> Running Full Training Pipeline (Integration Test)...")
    # run_training handles: Dataset init, DataLoader, Model init, Training Loop, Validation, Submission
    # We limit to 4 samples and 1 epoch for speed.
    try:
        run_training(limit=4, epochs=1)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # 6. Verify Submission Output
    print("\n>>> Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file {Config.SUBMISSION_PATH} was not created."
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    assert "Id" in df_sub.columns, "Submission missing 'Id' column."
    assert "Predicted" in df_sub.columns, "Submission missing 'Predicted' column."
    assert len(df_sub) > 0, "Submission file is empty."

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
