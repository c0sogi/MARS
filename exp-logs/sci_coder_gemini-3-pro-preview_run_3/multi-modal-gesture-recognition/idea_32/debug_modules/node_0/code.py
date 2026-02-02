import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import json

# Import the provided library modules
from library import config, utils, data_loader, model, trainer, predict


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo, overriding config paths
    and creating a subset of the metadata to ensure speed.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    demo_metadata_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_metadata_dir, exist_ok=True)

    demo_cache_dir = os.path.join(demo_dir, "cache")
    os.makedirs(demo_cache_dir, exist_ok=True)

    demo_submission_dir = os.path.join(demo_dir, "submission")
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Override config parameters for speed
    config.WORKING_DIR = demo_dir
    config.CACHE_DIR = demo_cache_dir
    config.SUBMISSION_DIR = demo_submission_dir
    config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")
    config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")

    # Reduce training parameters
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    # Create subset metadata
    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take top N samples
    subset_size = 8
    demo_train = orig_train.head(subset_size)
    demo_val = orig_val.head(subset_size)
    demo_test = orig_test.head(subset_size)

    # Save subset metadata
    config.TRAIN_METADATA_PATH = os.path.join(demo_metadata_dir, "train.csv")
    config.VAL_METADATA_PATH = os.path.join(demo_metadata_dir, "val.csv")
    config.TEST_METADATA_PATH = os.path.join(demo_metadata_dir, "test.csv")

    demo_train.to_csv(config.TRAIN_METADATA_PATH, index=False)
    demo_val.to_csv(config.VAL_METADATA_PATH, index=False)
    demo_test.to_csv(config.TEST_METADATA_PATH, index=False)

    print(f"Created subset metadata with {subset_size} samples each.")
    print("Config parameters overridden for demo.")


def test_utils():
    """
    Verifies utility functions.
    """
    print("\n=== Testing Utils ===")

    # 1. Levenshtein Distance
    # Sequence A: [1, 2, 3]
    # Sequence B: [1, 3] (Deletion of 2) -> Dist 1
    dist = utils.levenshtein_distance([1, 2, 3], [1, 3])
    assert dist == 1, f"Expected Levenshtein distance 1, got {dist}"

    # Substitution: [1, 2] vs [1, 3] -> Dist 1
    dist = utils.levenshtein_distance([1, 2], [1, 3])
    assert dist == 1, f"Expected Levenshtein distance 1, got {dist}"

    print("Levenshtein distance logic verified.")

    # 2. RLE and Sequence Processing
    # Input: [1, 1, 1, 0, 0, 2, 2, 2] -> (1,3), (0,2), (2,3)
    # Filter min_duration=3 -> (1,3), (2,3) (0 is removed if we treat it as noise or background)
    # The utils.process_gesture_sequence function removes background (ID 0) and merges.

    raw_preds = [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2]  # 1x5, 0x2, 2x5
    # If min_duration=3, 0 is dropped (len 2).
    # Result should be [1, 2]

    seq = utils.process_gesture_sequence(raw_preds, min_duration=3, background_id=0)
    assert seq == [1, 2], f"Expected sequence [1, 2], got {seq}"

    print("Sequence processing logic verified.")


def test_data_loading():
    """
    Verifies data loading and batch shapes.
    """
    print("\n=== Testing Data Loader ===")

    # Initialize loaders (this will process the subset metadata and cache it)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        load_cached_data=False
    )

    assert len(train_loader) > 0, "Train loader is empty"

    # Fetch one batch
    features, labels = next(iter(train_loader))

    # Expected Shapes:
    # Features: (Batch, WindowSize, InputDim)
    # InputDim = (20 joints * 9 channels) + 13 MFCC = 193
    expected_input_dim = (20 * 9) + 13

    print(f"Batch Features Shape: {features.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    assert (
        features.shape[0] == config.BATCH_SIZE
    ), f"Expected batch size {config.BATCH_SIZE}, got {features.shape[0]}"
    assert (
        features.shape[1] == config.WINDOW_SIZE
    ), f"Expected window size {config.WINDOW_SIZE}, got {features.shape[1]}"
    assert (
        features.shape[2] == expected_input_dim
    ), f"Expected input dim {expected_input_dim}, got {features.shape[2]}"
    assert labels.shape == (
        config.BATCH_SIZE,
        config.WINDOW_SIZE,
    ), "Labels shape mismatch"

    print("Data loader verification passed.")
    return features, labels


def test_model(sample_batch):
    """
    Verifies model initialization and forward pass.
    """
    print("\n=== Testing Model ===")

    features, labels = sample_batch
    device = config.DEVICE

    net = model.GHCMN().to(device)
    features = features.to(device)

    # Forward pass
    outputs = net(features)

    # Check outputs
    # We expect dictionary with stage1_logits, stage2_logits, stage3_logits, etc.
    required_keys = ["stage1_logits", "stage2_logits", "stage3_logits", "stage3_probs"]
    for k in required_keys:
        assert k in outputs, f"Model output missing key: {k}"

    # Check shape of logits: (Batch, Time, Classes)
    logits = outputs["stage3_logits"]
    assert logits.shape == (
        config.BATCH_SIZE,
        config.WINDOW_SIZE,
        config.NUM_CLASSES,
    ), f"Logits shape mismatch: {logits.shape}"

    print("Model forward pass verified.")
    return net


def test_training():
    """
    Verifies the training loop using the Trainer class.
    """
    print("\n=== Testing Trainer ===")

    # Instantiate Trainer
    # Note: Trainer re-initializes loaders internally, but since we overrode config paths,
    # it will pick up the subset data.
    t = trainer.Trainer()

    # Run training
    # We set epochs to 2 in setup, so this should be fast.
    t.train()

    # Check if model checkpoint was saved
    assert os.path.exists(
        config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("Training loop completed and model saved.")

    return t


def test_inference(trainer_instance):
    """
    Verifies inference and submission generation.
    """
    print("\n=== Testing Inference ===")

    # Generate predictions
    trainer_instance.generate_submission()

    # Check submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created."

    # Validate content format
    with open(config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Submission file is empty"

    # Check first line format: SessionID,label,label...
    first_line = lines[0].strip()
    parts = first_line.split(",")

    # First part should be session ID (starts with Sample or Session, or whatever the ID is)
    # Based on metadata, IDs are like 'Sample00300'
    assert len(parts) >= 1, "Invalid submission line format"
    print(f"Sample submission line: {first_line}")

    print("Inference pipeline verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Test Utils
        test_utils()

        # 3. Test Data Loading
        batch_features, batch_labels = test_data_loading()

        # 4. Test Model
        test_model((batch_features, batch_labels))

        # 5. Test Training
        trainer_instance = test_training()

        # 6. Test Inference
        test_inference(trainer_instance)

        print("\nALL DEMO TESTS PASSED SUCCESSFULLY.")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
