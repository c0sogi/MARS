import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import RNADataset
from library.model import DeepStabilizedBiGRU
from library.engine import train_and_evaluate, generate_submission, set_seed


def setup_demo_config():
    """
    Overrides Config parameters to run a fast demonstration.
    """
    # Create a specific directory for this demo execution
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Configuring demo run in: {demo_dir}")

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Override Cache paths to ensure we don't use existing full caches
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_cache.npy")

    # Override Training Hyperparameters for speed
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.CONV_FILTERS = 32  # Smaller stem
    Config.NUM_LAYERS = 2  # Fewer layers
    Config.PATIENCE = 2  # Early stopping

    # Ensure reproducibility
    set_seed(Config.SEED)


def verify_data_pipeline():
    """
    Verifies that the RNADataset loads data correctly and produces expected shapes.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Load a tiny subset of training data
    ds = RNADataset(split="train", load_cached_data=False, max_samples=32)

    print(f"Dataset length: {len(ds)}")
    assert (
        len(ds) == 32
    ), "Dataset should have 32 samples due to max_samples constraint."

    # Fetch one sample
    sample = ds[0]

    # Check keys
    required_keys = ["features", "adjacency", "id", "targets"]
    for k in required_keys:
        assert k in sample, f"Sample missing key: {k}"

    # Check Features Shape: (Seq_Len=107, Channels=14)
    features = sample["features"]
    assert features.shape == (107, 14), f"Unexpected features shape: {features.shape}"

    # Check Adjacency Shape: (Seq_Len=107)
    adjacency = sample["adjacency"]
    assert adjacency.shape == (107,), f"Unexpected adjacency shape: {adjacency.shape}"

    # Check Targets Shape: (Scored_Len=68, Num_Targets=5)
    targets = sample["targets"]
    assert targets.shape == (68, 5), f"Unexpected targets shape: {targets.shape}"

    print("Data pipeline verification passed.")
    return ds


def verify_model_architecture(dataset):
    """
    Verifies that the model instantiates and processes a batch correctly.
    """
    print("\n=== Verifying Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = DeepStabilizedBiGRU().to(device)

    # Create a batch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    features = batch["features"].to(device)
    adjacency = batch["adjacency"].to(device)

    # Forward pass
    outputs = model(features, adjacency)

    # Check Output Shape: (Batch=4, Seq_Len=107, Num_Targets=5)
    expected_shape = (4, 107, 5)
    assert (
        outputs.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {outputs.shape}"

    print("Model architecture verification passed.")


def run_demo_training():
    """
    Runs the training loop using the engine.
    """
    print("\n=== Running Demo Training ===")

    # We use max_samples=64 to simulate a very small dataset for quick training
    # This will trigger the train_and_evaluate function which saves the best model to Config.MODEL_PATH
    train_and_evaluate(load_cached_data=False, max_samples=64)

    # Verify model file was created
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file was not created at {Config.MODEL_PATH}")

    print("Demo training completed successfully.")


def run_demo_inference():
    """
    Runs inference using the trained model and generates a submission.
    """
    print("\n=== Running Demo Inference ===")

    # Generate submission
    # Note: generate_submission loads the model from Config.MODEL_PATH
    # It runs on the full test set (240 samples), which is fast.
    generate_submission(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Check submission content
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df.shape}")

    # Expected rows: 240 samples * 107 positions = 25680 rows
    # Note: The provided sample_submission.csv has 25681 lines (header + 25680 rows).
    expected_rows = 240 * 107
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df.columns) == expected_cols
    ), f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}"

    print("Demo inference and submission verification passed.")


if __name__ == "__main__":
    # 1. Configure
    setup_demo_config()

    # 2. Verify Data
    # We keep the dataset object to reuse for model verification
    train_ds = verify_data_pipeline()

    # 3. Verify Model
    verify_model_architecture(train_ds)

    # 4. Train
    run_demo_training()

    # 5. Inference
    run_demo_inference()

    print("\nAll demonstration steps completed successfully.")
