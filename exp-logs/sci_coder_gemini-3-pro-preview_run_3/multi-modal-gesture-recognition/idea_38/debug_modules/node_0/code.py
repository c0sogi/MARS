import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.data_loader import get_data_loaders
from library.trainer import Trainer, SHCGKN
from library.utils import (
    decode_predictions,
    calculate_levenshtein_distance,
    compute_dataset_metrics,
    run_length_encoding,
)


def setup_demo_config():
    """
    Overrides the default configuration to run a fast, lightweight demo.
    """
    # Set fixed seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define demo paths
    demo_work_dir = "./working/demo_execution"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir, exist_ok=True)

    # Patch the Config class attributes directly
    Config.WORKING_DIR = demo_work_dir
    Config.SUBMISSION_DIR = os.path.join(demo_work_dir, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(demo_work_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set params for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.WINDOW_SIZE = 32  # Smaller window for demo
    Config.STRIDE_TRAIN = 16
    Config.STRIDE_TEST = 16

    print(f"Configuration updated for demo execution at: {Config.WORKING_DIR}")


def test_data_loading():
    """
    Verifies that the data loader correctly processes input files and yields batches.
    """
    print("\n>>> Testing Data Loaders...")

    # Initialize loaders
    # Note: This will trigger caching of the debug subset
    train_loader, val_loader, test_loader = get_data_loaders(
        Config, load_cached_data=False
    )

    # Fetch one batch
    features, labels = next(iter(train_loader))

    # Validation
    # Features shape: (Batch, Window, InputDim)
    # Labels shape: (Batch, Window)
    print(f"  Batch Features Shape: {features.shape}")
    print(f"  Batch Labels Shape: {labels.shape}")

    expected_dim = Config.INPUT_DIM
    assert (
        features.shape[2] == expected_dim
    ), f"Expected input dim {expected_dim}, got {features.shape[2]}"
    assert (
        features.shape[1] == Config.WINDOW_SIZE
    ), f"Expected window size {Config.WINDOW_SIZE}, got {features.shape[1]}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
    ), "Labels shape mismatch"

    print("  Data Loader logic verified.")
    return train_loader


def test_model_architecture(train_loader):
    """
    Verifies the SHCGKN model instantiation and forward pass.
    """
    print("\n>>> Testing Model Architecture...")

    device = Config.get_device()
    model = SHCGKN(Config).to(device)
    model.eval()

    # Get a batch
    features, _ = next(iter(train_loader))
    features = features.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(features)

    # Check outputs
    required_keys = ["logits1", "probs1", "logits2", "probs2", "logits3", "probs3"]
    for key in required_keys:
        assert key in outputs, f"Model output missing key: {key}"

    # Check shape of final probabilities: (Batch, Window, NumClasses)
    probs3 = outputs["probs3"]
    print(f"  Output Probs Shape: {probs3.shape}")

    assert probs3.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NUM_CLASSES)}, got {probs3.shape}"

    print("  Model architecture verified.")


def test_utilities():
    """
    Verifies metric calculation and decoding logic.
    """
    print("\n>>> Testing Utilities...")

    # 1. Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = calculate_levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Distance should be 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2, 4]
    dist_diff = calculate_levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Distance should be 1 for one substitution, got {dist_diff}"

    print(
        f"  Levenshtein Distance logic verified (0 for identical, {dist_diff} for diff)."
    )

    # 2. Decoding Logic (Probs -> Sequence)
    # Create synthetic probabilities for 20 frames
    # Frames 0-9: Class 1
    # Frames 10-19: Class 2
    # Assuming MIN_GESTURE_DURATION is small enough (e.g. 5)

    num_frames = 20
    num_classes = Config.NUM_CLASSES
    fake_probs = np.zeros((num_frames, num_classes))

    # Set high probability for class 1
    fake_probs[0:10, 1] = 1.0
    # Set high probability for class 2
    fake_probs[10:20, 2] = 1.0

    # Decode
    # Note: decode_predictions applies argmax -> RLE -> Filter -> Sequence
    sequence = decode_predictions(fake_probs)

    print(f"  Decoded Sequence: {sequence}")

    # We expect [1, 2] assuming duration > min_duration
    # Config.MIN_GESTURE_DURATION is 5 by default
    assert sequence == [1, 2], f"Decoding failed. Expected [1, 2], got {sequence}"

    print("  Decoding logic verified.")


def run_training_pipeline():
    """
    Runs the full Trainer to demonstrate training loop and submission generation.
    """
    print("\n>>> Running Full Training Pipeline (1 Epoch)...")

    trainer = Trainer(Config)

    # Run training
    # This calls train_epoch, validate, and finally generate_submission
    trainer.run()

    # Verify artifacts
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model checkpoint was not saved.")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    # Check submission content
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    print(f"  Submission generated with {len(lines)} lines.")

    # We expect at least some lines (DEBUG_SUBSET_SIZE is 10, so 10 lines)
    # Note: If the test set in metadata is smaller than subset size, it might be fewer.
    # But we assert file is not empty.
    assert len(lines) > 0, "Submission file is empty."

    # Check format of first line: "SampleID,Label1,Label2..."
    parts = lines[0].strip().split(",")
    print(f"  Sample Line: {lines[0].strip()}")
    assert len(parts) >= 1, "Invalid submission format."

    print("  Training pipeline verified.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Data Loading
    loader = test_data_loading()

    # 3. Model
    test_model_architecture(loader)

    # 4. Utils
    test_utilities()

    # 5. Full Run
    run_training_pipeline()

    print("\n>>> All demonstrations completed successfully.")
