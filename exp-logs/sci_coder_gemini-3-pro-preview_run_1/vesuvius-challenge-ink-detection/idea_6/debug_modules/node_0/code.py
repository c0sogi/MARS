import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import (
    seed_everything,
    rle_encode,
    calculate_fbeta,
    find_best_threshold,
)
from library.dataset import InkDataset
from library.model import FFDCNet
from library.train import train_model
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Overrides Config parameters for a quick demo run and creates mini datasets.
    """
    print("--- Setting up Demo Environment ---")

    # 1. Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.PREDICTION_DIR = os.path.join(Config.WORKING_DIR, "predictions")

    # Use a separate submission file for the demo to avoid overwriting the main one immediately
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Reduce compute requirements
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny datasets

    # Re-run setup to create these new directories
    Config.setup()

    # 2. Create Mini Metadata Files (Subset of original data)
    # We take the first few rows of the existing metadata to create a tiny dataset

    # Train
    df_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    df_train_mini = df_train.head(4).copy()  # 4 samples = 2 batches
    mini_train_path = os.path.join(Config.WORKING_DIR, "train_mini.csv")
    df_train_mini.to_csv(mini_train_path, index=False)
    Config.TRAIN_METADATA = mini_train_path

    # Val
    df_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    df_val_mini = df_val.head(4).copy()
    mini_val_path = os.path.join(Config.WORKING_DIR, "val_mini.csv")
    df_val_mini.to_csv(mini_val_path, index=False)
    Config.VAL_METADATA = mini_val_path

    # Test
    df_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    df_test_mini = df_test.head(4).copy()
    mini_test_path = os.path.join(Config.WORKING_DIR, "test_mini.csv")
    df_test_mini.to_csv(mini_test_path, index=False)
    Config.TEST_METADATA = mini_test_path

    print("Mini datasets created.")
    print(f"Working Directory: {Config.WORKING_DIR}")


def verify_utils():
    """
    Verifies correctness of utility functions.
    """
    print("\n--- Verifying Utils ---")

    # 1. RLE Encode Verification
    # Create a simple mask: 0 1 1 0 1 (Indices: 1 2 3 4 5)
    # 1s are at indices 2, 3 and 5.
    # Run 1: Start 2, Length 2. Run 2: Start 5, Length 1.
    # Expected RLE: "2 2 5 1"
    mask = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)  # Flattened: 0 1 1 0 1 0
    rle_output = rle_encode(mask)
    expected_rle = "2 2 5 1"

    if rle_output != expected_rle:
        raise AssertionError(
            f"RLE Encode failed. Expected '{expected_rle}', got '{rle_output}'"
        )
    print("RLE Encode: OK")

    # 2. F-Beta Score Verification
    # Beta = 0.5 (Precision weighted higher)
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    # TP=1, FP=1, FN=1
    # Precision = 1/2 = 0.5
    # Recall = 1/2 = 0.5
    # F0.5 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    score = calculate_fbeta(y_true, y_pred, beta=0.5)

    if not np.isclose(score, 0.5):
        raise AssertionError(f"F-Beta calculation failed. Expected 0.5, got {score}")
    print("F-Beta Score: OK")


def verify_model_architecture():
    """
    Verifies model instantiation and forward pass.
    """
    print("\n--- Verifying Model Architecture ---")

    device = Config.DEVICE
    model = FFDCNet().to(device)

    # Create dummy input: (Batch, Depth, Height, Width)
    # Depth=65, H=512, W=512
    dummy_input = torch.randn(2, 65, 512, 512).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch, 1, Height, Width) - Logits
    expected_shape = (2, 1, 512, 512)

    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print(f"Model Forward Pass: OK (Output shape: {output.shape})")


def run_training_demo():
    """
    Runs the training loop using the mini dataset.
    """
    print("\n--- Running Training Demo ---")

    # Run training
    # load_cached_data=False forces the Dataset to load from raw TIFFs once, verifying that logic.
    model, best_score = train_model(load_cached_data=False)

    # Verify outputs
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    threshold_path = os.path.join(Config.WORKING_DIR, "best_threshold.txt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Training failed to save 'best_model.pth'.")

    if not os.path.exists(threshold_path):
        raise FileNotFoundError("Training failed to save 'best_threshold.txt'.")

    print(f"Training completed. Best F0.5: {best_score:.4f}")


def run_inference_demo():
    """
    Runs the inference loop and generates submission.csv.
    """
    print("\n--- Running Inference Demo ---")

    # Run inference
    generate_submission(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(f"Inference failed to create {Config.SUBMISSION_PATH}")

    # Verify content format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = ["Id", "Predicted"]

    if not all(col in df.columns for col in required_cols):
        raise AssertionError(
            f"Submission file missing required columns. Found: {df.columns}"
        )

    if len(df) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"Inference completed. Submission generated at {Config.SUBMISSION_PATH}")
    print("Sample Output:")
    print(df.head())

    # Copy to the required home directory location for the final check
    final_submission_path = "./submission.csv"
    shutil.copy(Config.SUBMISSION_PATH, final_submission_path)
    print(f"Copied submission to final location: {final_submission_path}")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Setup
    setup_demo_environment()

    # 2. Unit Verification
    verify_utils()
    verify_model_architecture()

    # 3. Pipeline Execution
    run_training_demo()
    run_inference_demo()

    print("\n=== Demo Completed Successfully ===")
