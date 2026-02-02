import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import random

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library import utils
from library import model
from library import data_loader
from library import train
from library import inference


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_environment():
    """
    Creates a lightweight environment for the demo by:
    1. Defining a temporary working directory.
    2. Creating mini-metadata files (subset of real data) for speed.
    3. Overriding config paths to use these mini-files.
    """
    print("Setting up demo environment...")

    # Define paths
    demo_dir = "./working/demo_env"
    cache_dir = os.path.join(demo_dir, "cache")
    meta_dir = os.path.join(demo_dir, "metadata")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 1. Override Config Constants
    config.BATCH_SIZE = 2
    config.CACHE_DIR = cache_dir
    config.NUM_EPOCHS = 2

    # 2. Create Mini Metadata (5 samples each) to avoid processing full dataset
    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Slice top 5
    mini_train = orig_train.head(5)
    mini_val = orig_val.head(5)
    mini_test = orig_test.head(5)

    # Save mini metadata
    mini_train_path = os.path.join(meta_dir, "train.csv")
    mini_val_path = os.path.join(meta_dir, "val.csv")
    mini_test_path = os.path.join(meta_dir, "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Override Config Paths
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    # Override Model Save Path to avoid overwriting main work
    config.WORKING_DIR = (
        demo_dir  # Best model will be saved to demo_dir/idea_34/best_model.pth
    )
    os.makedirs(os.path.join(demo_dir, "idea_34"), exist_ok=True)

    print(f"Demo environment configured. Using {len(mini_train)} training samples.")


def verify_utils():
    """Verifies utility functions logic."""
    print("\nVerifying Utils...")

    # Test Levenshtein
    # Distance between [1, 2] and [1, 3] is 1 (substitution)
    dist = utils.levenshtein_distance([1, 2], [1, 3])
    assert dist == 1, f"Expected Levenshtein distance 1, got {dist}"

    # Distance between [1] and [1, 2] is 1 (insertion)
    dist = utils.levenshtein_distance([1], [1, 2])
    assert dist == 1, f"Expected Levenshtein distance 1, got {dist}"

    # Test Run Length Encoding
    # [0, 0, 1, 1, 1, 1, 1, 0, 2, 2] -> Filter 0, keep 1 (len 5), filter 2 (len 2 < min 5)
    # Assuming MIN_GESTURE_DURATION is default 5
    preds = [0, 0, 1, 1, 1, 1, 1, 0, 2, 2]
    decoded = utils.run_length_encoding(preds, min_duration=5)
    assert decoded == [1], f"Expected [1], got {decoded}"

    print("Utils verification passed.")


def verify_model_architecture():
    """Verifies model instantiation and forward pass shapes."""
    print("\nVerifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for shape check to be quick
    net = model.PG_HCKN().to(device)

    # Create dummy input: (Batch, Time, Channels)
    # Channels = 180 (Skeleton) + 13 (Audio) = 193
    B, T, C = 2, 64, 193
    dummy_input = torch.randn(B, T, C).to(device)

    # Forward pass
    outputs = net(dummy_input)

    # Check outputs
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    # Check shape: (B, T, NumClasses)
    expected_shape = (B, T, config.NUM_CLASSES)
    assert (
        outputs["stage3"].shape == expected_shape
    ), f"Expected shape {expected_shape}, got {outputs['stage3'].shape}"

    print("Model architecture verification passed.")


def run_training_demo():
    """Runs a short training loop."""
    print("\nRunning Training Demo...")

    # Train for 2 epochs
    # Note: We rely on the overridden config paths pointing to mini datasets
    best_score = train.train_model(limit=None, epochs=2)

    print(f"Training demo finished. Best Validation Score: {best_score:.4f}")

    # Check if model file was created
    model_path = os.path.join(config.WORKING_DIR, "idea_34", "best_model.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print("Model checkpoint saved successfully.")
    return model_path


def run_inference_demo(model_path):
    """Runs inference using the trained model."""
    print("\nRunning Inference Demo...")

    output_csv = os.path.join(config.WORKING_DIR, "submission", "submission.csv")

    # Run inference
    inference.generate_submission(
        model_path=model_path, output_path=output_csv, limit=None
    )

    # Verify output
    assert os.path.exists(output_csv), "Submission CSV not generated."

    df = pd.read_csv(output_csv, header=None)
    # We expect 5 rows because we used mini_test.csv with 5 samples
    # Note: generate_submission writes lines. If header is not handled in generate_submission,
    # pandas read_csv might treat first row as header.
    # The provided inference code does NOT write a header row (it writes "sid,labels").
    # So we expect 5 rows.
    assert len(df) == 5, f"Expected 5 predictions, got {len(df)}"

    print(f"Inference demo finished. Output saved to {output_csv}")
    print("Sample Output:")
    print(df.head())


if __name__ == "__main__":
    try:
        set_seeds(42)

        # 1. Setup Environment (Mini Datasets)
        setup_demo_environment()

        # 2. Verify Components
        verify_utils()
        verify_model_architecture()

        # 3. Run Training Pipeline
        trained_model_path = run_training_demo()

        # 4. Run Inference Pipeline
        run_inference_demo(trained_model_path)

        print("\n=== All System Checks Passed Successfully ===")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
