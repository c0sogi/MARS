import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config, set_seed
from library.data import preprocess_data, ManufacturingDataset
from library.model import GatedFunnelNetwork
from library.train_eval import train_model
from library.utils import compute_auc


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo.
    Creates mini-datasets and overrides Config parameters for speed.
    """
    print("--- Setting up Demo Environment ---")

    # Define temporary paths
    demo_dir = "./working/demo_execution"
    mini_data_dir = os.path.join(demo_dir, "mini_data")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(mini_data_dir, exist_ok=True)

    # Override Config to use this directory
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir  # data.py uses WORKING_DIR for cache, but good to be explicit if needed
    Config.SUBMISSION_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Training Hyperparameters for Speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.HIDDEN_LAYERS = [64, 32]  # Smaller model
    Config.PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Create Mini Datasets (2000 rows each)
    print("Creating mini-datasets...")
    nrows = 2001  # Header + 2000 rows

    # Read original metadata files
    # Note: Using the metadata files provided in the problem description
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    df_train = pd.read_csv(orig_train_path, nrows=nrows)
    df_val = pd.read_csv(orig_val_path, nrows=nrows)
    df_test = pd.read_csv(orig_test_path, nrows=nrows)

    # Save mini files
    mini_train_path = os.path.join(mini_data_dir, "train.csv")
    mini_val_path = os.path.join(mini_data_dir, "val.csv")
    mini_test_path = os.path.join(mini_data_dir, "test.csv")

    df_train.to_csv(mini_train_path, index=False)
    df_val.to_csv(mini_val_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    # Update Config paths to point to mini datasets
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path

    print("Demo environment setup complete.\n")


def verify_data_pipeline():
    """
    Verifies the data loading and preprocessing pipeline.
    """
    print("--- Verifying Data Pipeline ---")

    # Run preprocessing
    # This will generate cache files in the demo directory
    train_loader, val_loader, test_loader, vocab_sizes, num_cont = preprocess_data(
        load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    # Assertions
    print("Checking vocab sizes and feature counts...")
    assert isinstance(vocab_sizes, np.ndarray), "vocab_sizes should be a numpy array"
    assert len(vocab_sizes) > 0, "vocab_sizes should not be empty"
    assert num_cont > 0, "Number of continuous features should be positive"

    print("Checking Train Loader...")
    batch = next(iter(train_loader))
    assert len(batch) == 3, "Train loader should yield (cat, cont, target)"
    x_cat, x_cont, y = batch

    assert (
        x_cat.shape[0] == Config.BATCH_SIZE or x_cat.shape[0] <= 2000
    ), "Batch size mismatch"
    assert x_cat.shape[1] == len(vocab_sizes), "Categorical feature dimension mismatch"
    assert x_cont.shape[1] == num_cont, "Continuous feature dimension mismatch"
    assert y.shape[1] == 1, "Target shape mismatch"

    print("Checking Test Loader...")
    test_batch = next(iter(test_loader))
    assert len(test_batch) == 2, "Test loader should yield (cat, cont)"

    print("Data Pipeline verification passed.\n")
    return vocab_sizes, num_cont, train_loader


def verify_model_logic(vocab_sizes, num_cont, train_loader):
    """
    Verifies model instantiation and forward pass.
    """
    print("--- Verifying Model Logic ---")

    # Instantiate Model
    model = GatedFunnelNetwork(vocab_sizes, num_cont, Config)
    model.to(Config.DEVICE)
    model.eval()

    # Get a batch
    x_cat, x_cont, y = next(iter(train_loader))
    x_cat = x_cat.to(Config.DEVICE)
    x_cont = x_cont.to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        logits = model(x_cat, x_cont)

    # Assertions
    print("Checking output shape...")
    assert logits.shape == (
        x_cat.size(0),
        1,
    ), f"Expected output shape {(x_cat.size(0), 1)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model verification passed.\n")


def verify_metrics():
    """
    Verifies the metric computation utility.
    """
    print("--- Verifying Metrics ---")
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])

    auc = compute_auc(y_true, y_scores)
    print(f"Computed AUC: {auc}")
    assert 0 <= auc <= 1, "AUC must be between 0 and 1"

    print("Metric verification passed.\n")


def run_full_training_demo():
    """
    Runs the full training loop using the library function.
    """
    print("--- Running Full Training Pipeline ---")

    # We set load_cached_data=True because verify_data_pipeline already created the cache
    # in the demo working directory.
    train_model(load_cached_data=True, epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # Verify outputs
    print("Verifying output files...")
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated"

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Full training pipeline finished successfully.\n")


if __name__ == "__main__":
    try:
        # 1. Setup Environment
        setup_demo_environment()

        # 2. Verify Data Processing
        vocab_sizes, num_cont, train_loader = verify_data_pipeline()

        # 3. Verify Model
        verify_model_logic(vocab_sizes, num_cont, train_loader)

        # 4. Verify Metrics
        verify_metrics()

        # 5. Run Full Training Loop
        run_full_training_demo()

        print("All demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
