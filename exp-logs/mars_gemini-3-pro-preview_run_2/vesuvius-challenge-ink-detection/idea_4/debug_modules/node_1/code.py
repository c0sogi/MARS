import os
import shutil
import torch
import numpy as np
import pandas as pd
import cv2

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, fbeta_score, dice_coef
from library.data import InkDataset, get_loaders
from library.model import HPUnet
from library.train import train_model
from library.inference import run_inference


def test_utils():
    """Verifies utility functions."""
    print("\n--- Testing Utils ---")

    # 1. Test RLE Encoding
    # Pattern: [1, 0, 0, 0, 1] -> Indices 1 and 5 (1-based)
    # Expected: Start 1 Len 1, Start 5 Len 1 -> "1 1 5 1"
    dummy_mask = np.array([[1, 0, 0, 0, 1]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    print(f"RLE Input: [1, 0, 0, 0, 1] -> Encoded: '{encoded}'")
    assert (
        encoded == "1 1 5 1"
    ), f"RLE Encoding failed. Expected '1 1 5 1', got '{encoded}'"

    # 2. Test Metrics
    # Preds: 0.8 (Ink), 0.2 (Bg) -> Binary: 1, 0
    # Targets: 1 (Ink), 0 (Bg)
    preds = torch.tensor([0.8, 0.2])
    targets = torch.tensor([1.0, 0.0])

    f05 = fbeta_score(preds, targets, beta=0.5, threshold=0.5)
    dice = dice_coef(preds, targets, threshold=0.5)

    print(f"Metrics Check: F0.5={f05}, Dice={dice}")
    assert np.isclose(f05, 1.0), f"F0.5 calculation failed. Got {f05}"
    assert np.isclose(dice, 1.0), f"Dice calculation failed. Got {dice}"

    print("Utils verified.")


def test_data_pipeline():
    """Verifies dataset loading and shapes."""
    print("\n--- Testing Data Pipeline ---")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Use a tiny subset (first 4 patches)
    subset_df = train_df.iloc[:4].copy()

    # Initialize Dataset
    # load_cached_data=False ensures we test the raw volume processing logic
    ds = InkDataset(subset_df, phase="train", load_cached_data=False)

    print(f"Dataset initialized with {len(ds)} samples.")

    # Get one sample
    img, mask = ds[0]

    print(f"Sample Image Shape: {img.shape}")  # Expected: (4, 512, 512)
    print(f"Sample Mask Shape: {mask.shape}")  # Expected: (1, 512, 512)

    assert img.shape == (4, 512, 512), f"Image shape mismatch. Got {img.shape}"
    assert mask.shape == (1, 512, 512), f"Mask shape mismatch. Got {mask.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert isinstance(mask, torch.Tensor), "Mask is not a Tensor"

    print("Data pipeline verified.")


def test_model_architecture():
    """Verifies model instantiation and forward pass."""
    print("\n--- Testing Model Architecture ---")

    model = HPUnet(in_channels=4, classes=1)

    # Create dummy input: (Batch=2, Channels=4, H=512, W=512)
    dummy_input = torch.randn(2, 4, 512, 512)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (
        2,
        1,
        512,
        512,
    ), f"Model output shape mismatch. Got {output.shape}"
    print("Model architecture verified.")


def run_demo_execution():
    """Runs a complete training and inference cycle with reduced parameters."""
    print("\n--- Running Demo Training & Inference ---")

    # 1. Configure for Demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.WORKING_DIR = "./working/demo_run_script"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Run Training
    print("Step 1: Training...")
    # load_cached_data=False to ensure we test the generation logic
    best_score = train_model(load_cached_data=False)

    print(f"Training finished. Best F0.5: {best_score}")

    # Verify Checkpoint
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError("Checkpoint file was not created during training.")

    # 3. Run Inference
    print("Step 2: Inference...")
    run_inference(load_cached_data=False)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created during inference.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(sub_df.head())

    # Basic Validation
    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check if predictions look like RLE strings (numbers and spaces) or empty
    first_pred = str(sub_df.iloc[0]["Predicted"])
    if first_pred and first_pred != "nan":
        # It should be a string of space-separated numbers
        parts = first_pred.split()
        assert all(
            p.isdigit() for p in parts
        ), "Submission contains non-digit characters in RLE."

    print("Demo run completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Set a temporary working directory for the tests to avoid cluttering default paths
    Config.WORKING_DIR = "./working/demo_run_script"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    try:
        test_utils()
        test_data_pipeline()
        test_model_architecture()
        run_demo_execution()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        raise e
