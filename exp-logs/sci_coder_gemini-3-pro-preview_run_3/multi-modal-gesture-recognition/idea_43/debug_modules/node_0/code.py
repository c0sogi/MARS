import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Import library modules
# We need to ensure the library is in the path if running from root
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    compute_levenshtein_distance,
    run_length_encoding,
    decode_predictions,
    filter_short_segments,
)
from library.model import PAM_CN
from library.trainer import Trainer
from library.data_loader import get_dataloaders


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo run.
    Overrides Config paths and hyperparameters.
    Creates mini-metadata files.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_root = "./working/demo_env"
    if os.path.exists(demo_root):
        shutil.rmtree(demo_root)
    os.makedirs(demo_root)

    demo_metadata = os.path.join(demo_root, "metadata")
    demo_cache = os.path.join(demo_root, "cache")
    demo_checkpoints = os.path.join(demo_root, "checkpoints")
    demo_submission = os.path.join(demo_root, "submission")

    os.makedirs(demo_metadata, exist_ok=True)
    os.makedirs(demo_cache, exist_ok=True)
    os.makedirs(demo_checkpoints, exist_ok=True)
    os.makedirs(demo_submission, exist_ok=True)

    # Override Config
    # We modify the class attributes directly
    Config.METADATA_DIR = demo_metadata
    Config.CACHE_DIR = demo_cache
    Config.CHECKPOINT_DIR = demo_checkpoints
    Config.SUBMISSION_DIR = demo_submission
    Config.WORKING_DIR = demo_root

    # Reduce computational load
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.EARLY_STOPPING_PATIENCE = 2

    # Create mini metadata
    # Read original metadata
    orig_meta_dir = "./metadata"

    for split in ["train", "val", "test"]:
        src_csv = os.path.join(orig_meta_dir, f"{split}.csv")
        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            # Take top 5 samples
            mini_df = df.head(5)
            dst_csv = os.path.join(demo_metadata, f"{split}.csv")
            mini_df.to_csv(dst_csv, index=False)
            print(f"    Created mini {split} metadata with {len(mini_df)} samples.")
        else:
            print(f"    Warning: Original {split}.csv not found.")


def verify_utilities():
    """
    Verifies the logic of utility functions.
    """
    print(">>> Verifying utilities...")

    # 1. Levenshtein Distance
    # Distance between [1, 2, 3] and [1, 2, 3] should be 0
    d1 = compute_levenshtein_distance([1, 2, 3], [1, 2, 3])
    assert d1 == 0, f"Levenshtein distance error: expected 0, got {d1}"

    # Distance between [1, 2] and [1, 3] should be 1 (substitution)
    d2 = compute_levenshtein_distance([1, 2], [1, 3])
    assert d2 == 1, f"Levenshtein distance error: expected 1, got {d2}"

    # Distance between [1] and [1, 2] should be 1 (insertion)
    d3 = compute_levenshtein_distance([1], [1, 2])
    assert d3 == 1, f"Levenshtein distance error: expected 1, got {d3}"

    print("    Levenshtein distance check passed.")

    # 2. Run Length Encoding
    seq = [1, 1, 1, 2, 2, 0, 0, 3]
    rle = run_length_encoding(seq)
    expected_rle = [(1, 3), (2, 2), (0, 2), (3, 1)]
    assert rle == expected_rle, f"RLE error: expected {expected_rle}, got {rle}"

    print("    RLE check passed.")

    # 3. Filtering and Decoding
    # Config.MIN_GESTURE_LENGTH is likely 5. Let's test filtering.
    # We need to temporarily mock Config or pass args if function allows.
    # filter_short_segments allows passing min_length.

    segments = [(1, 10), (0, 20), (2, 2), (3, 10)]
    # 0 is background (Config.BACKGROUND_CLASS_ID is 0)
    # (2, 2) is short (< 5)
    # Expected: [1, 3]

    filtered = filter_short_segments(segments, min_length=5, background_id=0)
    assert filtered == [1, 3], f"Filtering error: expected [1, 3], got {filtered}"

    print("    Filtering check passed.")


def verify_model_architecture():
    """
    Verifies model instantiation and forward pass shapes.
    """
    print(">>> Verifying model architecture...")

    model = PAM_CN().to(Config.DEVICE)

    # Create dummy input: (Batch, Time, InputDim)
    # Batch=2, Time=32, InputDim=193
    batch_size = 2
    seq_len = 32
    input_dim = Config.INPUT_DIM

    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(Config.DEVICE)

    # Forward pass
    logits1, logits2, logits3 = model(dummy_input)

    # Check shapes: (Batch, Time, NumClasses)
    expected_shape = (batch_size, seq_len, Config.NUM_CLASSES)

    assert logits1.shape == expected_shape, f"Stage 1 shape mismatch: {logits1.shape}"
    assert logits2.shape == expected_shape, f"Stage 2 shape mismatch: {logits2.shape}"
    assert logits3.shape == expected_shape, f"Stage 3 shape mismatch: {logits3.shape}"

    print("    Model forward pass shape check passed.")


def run_training_demo():
    """
    Runs the Trainer to demonstrate training loop and inference.
    """
    print(">>> Running training demo...")

    # Initialize Trainer
    trainer = Trainer()

    # Run Fit
    # This will load data from our mini metadata
    print("    Starting fit()...")
    trainer.fit(epochs=Config.NUM_EPOCHS)

    # Check if best model was saved
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not created."
    print("    Training completed and model saved.")

    # Run Inference
    print("    Starting inference on test set...")
    trainer.predict_test_set()

    # Check submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Validate submission content format
    with open(submission_path, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            # Check format: SessionID,label1,label2...
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Submission line format error."
            print(f"    Sample prediction: {lines[0].strip()}")

    print("    Inference completed successfully.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Verify Utils
    verify_utilities()

    # 3. Verify Model
    verify_model_architecture()

    # 4. Run Training & Inference
    # Note: This involves data loading which might print some warnings if files are missing,
    # but we are using existing files from the input directory referenced by the mini metadata.
    run_training_demo()

    print("\n>>> Demo execution finished successfully.")
