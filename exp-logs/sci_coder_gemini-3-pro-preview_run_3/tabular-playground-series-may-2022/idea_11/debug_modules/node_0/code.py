import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.data_processing import get_dataloaders
from library.model import LayerNormFunnelMLP
from library.training import run_training


def setup_demo_config():
    """
    Overrides the default configuration to use a specific demo directory
    and faster training settings.
    """
    print("Setting up demo configuration...")

    # 1. Define new directories
    demo_dir = "./working/demo_execution"
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir

    # 2. Update derived paths manually (since they were evaluated at import time)
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.CACHE_PATH_TRAIN = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.CACHE_PATH_VAL = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.CACHE_PATH_TEST = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.METADATA_CACHE_PATH = os.path.join(Config.WORKING_DIR, "metadata.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # 3. Set hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4096  # Large batch size for A100 efficiency

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory set to: {Config.WORKING_DIR}")


def verify_model_architecture():
    """
    Unit test to verify the model accepts inputs and produces correct output shapes.
    """
    print("\nVerifying model architecture...")

    # Define dummy dimensions
    # Assume we have 2 categorical features (f_29, f_30) + 10 chars from f_27 = 12 categorical inputs
    # Let's assume vocab size of 50 for each for testing
    num_cat_features = 12
    vocab_sizes = [50] * num_cat_features

    # Continuous features: f_00..f_28 (excl 27) + unique_char_count = 29 features
    cont_dim = 29

    batch_size = 32

    # Create dummy tensors
    dummy_cat = torch.randint(0, 50, (batch_size, num_cat_features)).long()
    dummy_cont = torch.randn(batch_size, cont_dim).float()

    # Instantiate model
    model = LayerNormFunnelMLP(
        vocab_sizes=vocab_sizes,
        cont_dim=cont_dim,
        embed_dim=8,
        hidden_layers=[32, 16],
        token_dropout_rate=0.1,
        dropout_rate=0.1,
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_cat, dummy_cont)

    # Assertions
    assert logits.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {logits.shape}"

    print("Model architecture verification passed.")


def validate_submission_file(filepath):
    """
    Verifies the generated submission file format and content.
    """
    print(f"\nValidating submission file at {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Submission file not found at {filepath}")

    df = pd.read_csv(filepath)

    # Check shape (100,000 test samples + header)
    expected_rows = 100000
    if len(df) != expected_rows:
        raise AssertionError(f"Submission has {len(df)} rows, expected {expected_rows}")

    # Check columns
    expected_cols = ["id", "target"]
    if list(df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns are {list(df.columns)}, expected {expected_cols}"
        )

    # Check value types
    if not pd.api.types.is_numeric_dtype(df["id"]):
        raise AssertionError("ID column is not numeric")
    if not pd.api.types.is_numeric_dtype(df["target"]):
        raise AssertionError("Target column is not numeric")

    # Check probability range
    if df["target"].min() < 0 or df["target"].max() > 1:
        raise AssertionError("Target probabilities are out of range [0, 1]")

    print("Submission file validation passed.")


if __name__ == "__main__":
    # 1. Setup Configuration
    setup_demo_config()

    # 2. Verify Model Logic
    verify_model_architecture()

    # 3. Run Training Pipeline
    # We force load_cached_data=False to demonstrate the full processing pipeline
    # The processing is fast enough for this dataset size.
    print("\nStarting training pipeline execution...")
    trained_model = run_training(
        load_cached_data=False, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
    )

    # 4. Verify Outputs
    print("\nVerifying outputs...")

    # Check model file
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint found at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved.")

    # Check submission file
    validate_submission_file(Config.SUBMISSION_PATH)

    print("\nDemo execution completed successfully.")
