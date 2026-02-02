import os
import pandas as pd
import torch
import numpy as np
import shutil
import time

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_train_val_datasets, get_test_dataset
from library.model import MILEfficientNet
from library.train_eval import run_training, predict_and_submit


def create_mini_metadata(source_path, dest_path, n=4):
    """Creates a small subset of metadata for rapid demonstration."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Take top n rows
    df_subset = df.head(n).copy()
    df_subset.to_csv(dest_path, index=False)
    print(f"Created mini metadata at {dest_path} with {len(df_subset)} samples.")
    return len(df_subset)


def run_demonstration():
    print("=== Starting Library Demonstration ===\n")

    # 1. Configuration Overrides for Speed and Isolation
    # We modify the Config class attributes directly to isolate this run
    # and ensure it finishes quickly.
    print("--- Configuring Environment ---")
    Config.WORKING_DIR = "./working"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_execution")
    Config.MODEL_SAVE_PATH = os.path.join(Config.SUBMISSION_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create mini metadata files
    mini_train_path = os.path.join(Config.SUBMISSION_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.SUBMISSION_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.SUBMISSION_DIR, "mini_test.csv")

    # We use the original metadata provided in the environment to create subsets
    n_samples = create_mini_metadata("./metadata/train.csv", mini_train_path, n=4)
    create_mini_metadata("./metadata/val.csv", mini_val_path, n=4)
    create_mini_metadata("./metadata/test.csv", mini_test_path, n=4)

    # Point Config to these new mini files
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    # Reduce training parameters
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2

    # Set Seed
    set_seed(42)
    print("Configuration updated for demo run.\n")

    # 2. Data Loading Demonstration
    print("--- Testing Data Loader ---")
    # Force reload to ignore any existing cache and process our mini metadata
    train_ds, val_ds = get_train_val_datasets(load_cached=False)

    print(f"Train Dataset Length: {len(train_ds)}")
    print(f"Val Dataset Length: {len(val_ds)}")

    # Verify Dataset Logic
    assert len(train_ds) == n_samples, "Train dataset length mismatch."

    # Fetch one sample to verify tensor shapes
    # Expected: (Candidates, Channels, H, W) -> (3, 12, 224, 224)
    sample_data, sample_label, sample_id = train_ds[0]

    print(f"Sample Data Shape: {sample_data.shape}")
    print(f"Sample Label: {sample_label}")
    print(f"Sample ID: {sample_id}")

    assert sample_data.shape == (
        Config.NUM_CANDIDATES,
        Config.NUM_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected data shape: {sample_data.shape}"
    assert isinstance(sample_label.item(), float), "Label should be float."
    print("Data Loader verification successful.\n")

    # 3. Model Architecture Demonstration
    print("--- Testing Model Architecture ---")
    model = MILEfficientNet()
    model.eval()

    # Create a batch of size 2 using the sample data
    # Input expected: (Batch, Candidates, Channels, H, W)
    batch_input = torch.stack([sample_data, sample_data])
    print(f"Model Input Shape: {batch_input.shape}")

    with torch.no_grad():
        logits = model(batch_input)

    print(f"Model Output (Logits) Shape: {logits.shape}")

    # Verify Output Shape: (Batch, Candidates) -> (2, 3)
    assert logits.shape == (
        2,
        Config.NUM_CANDIDATES,
    ), f"Unexpected output shape: {logits.shape}"
    print("Model architecture verification successful.\n")

    # 4. Training Loop Demonstration
    print("--- Testing Training Loop ---")
    # This will run for 1 epoch on the mini dataset (4 samples, batch size 2 -> 2 steps)
    start_time = time.time()
    run_training(epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)
    end_time = time.time()

    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Verify Model Checkpoint
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file was not saved at {Config.MODEL_SAVE_PATH}"
    print(f"Verified model checkpoint exists at {Config.MODEL_SAVE_PATH}.\n")

    # 5. Inference Demonstration
    print("--- Testing Inference Pipeline ---")
    # This will generate predictions for the mini test set
    predict_and_submit(batch_size=Config.BATCH_SIZE)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Submission columns mismatch."
    assert (
        len(df_sub) == 4
    ), "Submission row count mismatch (expected 4 based on mini test)."
    assert (
        df_sub["BraTS21ID"].dtype == object or df_sub["BraTS21ID"].dtype == str
    ), "BraTS21ID should be treated as string/object to preserve leading zeros."

    # Check ID formatting (length 5)
    first_id = str(df_sub.iloc[0]["BraTS21ID"])
    assert len(first_id) == 5, f"BraTS21ID formatting incorrect: {first_id}"

    print("Inference verification successful.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
