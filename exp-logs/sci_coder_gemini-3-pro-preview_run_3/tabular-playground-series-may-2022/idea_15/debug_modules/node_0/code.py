import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders, process_data
from library.model import SwishFunnelNet
from library.train_utils import train_model, generate_submission


def setup_demo_environment():
    """
    Sets up a small subset of data and overrides Config to use it.
    This ensures the demo runs quickly.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_execution"
    mini_data_dir = os.path.join(demo_dir, "mini_data")

    os.makedirs(mini_data_dir, exist_ok=True)

    # Create mini datasets (first 2000 rows)
    # Reading from the metadata folder as per instructions
    print("Creating mini datasets...")
    pd.read_csv("./metadata/train.csv", nrows=2002).to_csv(
        os.path.join(mini_data_dir, "train.csv"), index=False
    )
    pd.read_csv("./metadata/val.csv", nrows=2002).to_csv(
        os.path.join(mini_data_dir, "val.csv"), index=False
    )
    pd.read_csv("./metadata/test.csv", nrows=2002).to_csv(
        os.path.join(mini_data_dir, "test.csv"), index=False
    )

    # Override Config paths to point to mini data
    Config.TRAIN_PATH = os.path.join(mini_data_dir, "train.csv")
    Config.VAL_PATH = os.path.join(mini_data_dir, "val.csv")
    Config.TEST_PATH = os.path.join(mini_data_dir, "test.csv")

    # Override working directory and cache paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_processed.parquet")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_processed.parquet")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_processed.parquet")
    Config.METADATA_CACHE = os.path.join(demo_dir, "metadata.npy")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.HIDDEN_LAYERS = [64, 32]  # Smaller model for demo
    Config.NUM_WORKERS = 2
    Config.EARLY_STOPPING_PATIENCE = 2

    print("Config overridden for demo execution.")


def test_data_pipeline():
    """
    Tests the data loading and processing pipeline.
    """
    print("\n=== Testing Data Pipeline ===")

    # Force reload from scratch (ignore cache if it exists from previous runs)
    if os.path.exists(Config.TRAIN_CACHE):
        os.remove(Config.TRAIN_CACHE)

    train_loader, val_loader, test_loader, num_continuous, vocab_sizes = (
        get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            load_cached_data=False,
        )
    )

    print(f"Num Continuous Features: {num_continuous}")
    print(f"Vocab Sizes: {vocab_sizes}")

    # Assertions
    assert num_continuous > 0, "Number of continuous features should be positive."
    assert len(vocab_sizes) > 0, "Should have categorical vocabulary sizes."
    assert len(train_loader) > 0, "Train loader should not be empty."

    # Check batch structure
    x_cont, x_cat, y = next(iter(train_loader))
    assert (
        x_cont.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {x_cont.shape[0]}"
    assert x_cont.shape[1] == num_continuous, "Continuous feature dimension mismatch."
    assert x_cat.shape[1] == len(vocab_sizes), "Categorical feature dimension mismatch."
    assert y.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch."

    print("Data Pipeline Verified.")
    return num_continuous, vocab_sizes


def test_model_architecture(num_continuous, vocab_sizes):
    """
    Tests the model instantiation and forward pass.
    """
    print("\n=== Testing Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for simple shape check

    model = SwishFunnelNet(
        num_continuous=num_continuous,
        categorical_vocab_sizes=vocab_sizes,
        embedding_dim=8,  # Small embedding for demo
        hidden_layers=[32, 16],
        dropout_rate=0.1,
        attn_bottleneck_dim=16,
        output_dim=1,
    ).to(device)

    # Create dummy input
    batch_size = 4
    dummy_cont = torch.randn(batch_size, num_continuous).to(device)
    dummy_cat = torch.zeros(batch_size, len(vocab_sizes), dtype=torch.long).to(device)

    # Forward pass
    logits = model(dummy_cont, dummy_cat)

    # Check output
    assert logits.shape == (
        batch_size,
        1,
    ), f"Output shape mismatch. Expected {(batch_size, 1)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    print("Model Architecture Verified.")


def test_training_loop():
    """
    Tests the full training loop using train_utils.
    """
    print("\n=== Testing Training Loop ===")

    # Ensure no previous model exists
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    # Run training
    # We use load_cached_data=True because we generated the cache in test_data_pipeline
    train_model(load_cached_data=True)

    # Verify model artifact
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    file_size = os.path.getsize(Config.MODEL_SAVE_PATH)
    assert file_size > 0, "Model file is empty."

    print(f"Training Loop Verified. Model saved at {Config.MODEL_SAVE_PATH}")


def test_inference_submission():
    """
    Tests the submission generation.
    """
    print("\n=== Testing Inference & Submission ===")

    # Ensure no previous submission exists
    if os.path.exists(Config.SUBMISSION_FILE):
        os.remove(Config.SUBMISSION_FILE)

    generate_submission(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert "target" in df_sub.columns, "Submission missing 'target' column."

    # Check length (should match the mini test set size)
    # We read 2002 rows for test.csv in setup_demo_environment
    expected_len = 2002
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    # Check value range
    assert df_sub["target"].min() >= 0.0, "Probabilities should be >= 0."
    assert df_sub["target"].max() <= 1.0, "Probabilities should be <= 1."

    print("Inference & Submission Verified.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Test Data Pipeline
        num_cont, vocab_sizes = test_data_pipeline()

        # 3. Test Model
        test_model_architecture(num_cont, vocab_sizes)

        # 4. Test Training
        test_training_loop()

        # 5. Test Submission
        test_inference_submission()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
