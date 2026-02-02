import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import library components
from library.config import Config
from library.utils import set_seed, levenshtein_distance, decode_predictions_to_gestures
from library.model import DW_AIIN
from library.train import run_training
from library.predict import run_prediction


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    set_seed(42)

    # Define demo directories
    demo_work_dir = "./working/demo_execution"
    demo_metadata_dir = os.path.join(demo_work_dir, "metadata")
    demo_cache_dir = os.path.join(demo_work_dir, "cache")
    demo_checkpoint_dir = os.path.join(demo_work_dir, "checkpoints")
    demo_submission_dir = os.path.join(demo_work_dir, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)

    os.makedirs(demo_metadata_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_checkpoint_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    print(f"Created demo working directory: {demo_work_dir}")

    # 2. Data Subsetting (Create mini-datasets for speed)
    # -------------------------------------------------------------------------
    print("Creating subset metadata for rapid demonstration...")

    # Helper to create subset
    def create_subset(src_path, dest_path, n=10):
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source metadata not found: {src_path}")
        df = pd.read_csv(src_path)
        # Take top N samples
        subset = df.head(n)
        subset.to_csv(dest_path, index=False)
        return len(subset)

    n_train = create_subset(
        Config.TRAIN_METADATA_PATH, os.path.join(demo_metadata_dir, "train.csv"), n=12
    )
    n_val = create_subset(
        Config.VAL_METADATA_PATH, os.path.join(demo_metadata_dir, "val.csv"), n=8
    )
    n_test = create_subset(
        Config.TEST_METADATA_PATH, os.path.join(demo_metadata_dir, "test.csv"), n=8
    )

    print(f"Subset sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # 3. Runtime Configuration Override
    # -------------------------------------------------------------------------
    print("Overriding Config parameters...")

    # Paths
    Config.WORK_DIR = demo_work_dir
    Config.CACHE_DIR = demo_cache_dir
    Config.CHECKPOINT_DIR = demo_checkpoint_dir
    Config.BEST_MODEL_PATH = os.path.join(demo_checkpoint_dir, "best_model.pth")
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    Config.TRAIN_METADATA_PATH = os.path.join(demo_metadata_dir, "train.csv")
    Config.VAL_METADATA_PATH = os.path.join(demo_metadata_dir, "val.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_metadata_dir, "test.csv")

    # Hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.EARLY_STOPPING_PATIENCE = 2

    # 4. Verify Utility Logic
    # -------------------------------------------------------------------------
    print("Verifying utility functions...")

    # Test Levenshtein Distance
    # Distance between [1, 2] and [1, 3] is 1 (substitution)
    d1 = levenshtein_distance([1, 2], [1, 3])
    assert d1 == 1, f"Levenshtein logic error: expected 1, got {d1}"

    # Distance between [1, 2] and [1, 2, 3] is 1 (insertion)
    d2 = levenshtein_distance([1, 2], [1, 2, 3])
    assert d2 == 1, f"Levenshtein logic error: expected 1, got {d2}"

    # Test Decoder
    # Input: [0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 0]
    # Min length 5.
    # 1s: length 5 -> Keep.
    # 2s: length 2 -> Discard (too short).
    # 0s: Background -> Discard.
    # Expected: [1]
    raw_preds = np.array([0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 0])
    decoded = decode_predictions_to_gestures(
        raw_preds, background_label=0, min_length=5
    )
    assert decoded == [1], f"Decoder logic error: expected [1], got {decoded}"

    print("Utilities verified.")

    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = DW_AIIN().to(device)

    # Create dummy inputs
    # Batch=2, Time=50
    # Skeleton: 60 channels
    # Audio: 13 channels
    B, T = 2, 50
    dummy_skel = torch.randn(B, T, 60).to(device)
    dummy_audio = torch.randn(B, T, 13).to(device)
    dummy_lengths = torch.tensor([50, 40]).to(device)  # Second sample padded

    # Forward pass
    outputs = model(dummy_skel, dummy_audio, dummy_lengths)

    # Expected Output: (B, T, NumClasses)
    expected_shape = (B, T, Config.NUM_CLASSES)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    print("Model forward pass successful.")

    # 6. Run Training Pipeline
    # -------------------------------------------------------------------------
    print("\n=== Running Training Pipeline ===")
    # This will use the subset data, calculate stats, cache it, and train for 2 epochs.
    run_training(num_epochs=Config.NUM_EPOCHS, load_cached_data=True)

    # Verify checkpoint creation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Training failed to produce 'best_model.pth'.")

    print("Training completed successfully.")

    # 7. Run Prediction Pipeline
    # -------------------------------------------------------------------------
    print("\n=== Running Prediction Pipeline ===")
    # This will load the best model and run inference on the subset test data.
    run_prediction(load_cached_data=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Prediction failed to produce 'submission.csv'.")

    # Check content
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    print(f"Submission file generated with {len(lines)} lines.")

    # We expect one line per test sample (n_test = 8)
    assert (
        len(lines) == n_test
    ), f"Expected {n_test} lines in submission, got {len(lines)}"

    print("Prediction pipeline completed successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
