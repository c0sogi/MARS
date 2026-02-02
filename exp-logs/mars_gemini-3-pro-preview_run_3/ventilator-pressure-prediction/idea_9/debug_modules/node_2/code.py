import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.model import RPCNet
from library.data_utils import get_dataloaders
from library.train_utils import run_training


def setup_demo_config():
    """
    Overrides the default configuration to run a fast demo.
    """
    print("Configuring environment for demo execution...")

    # 1. General Settings
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for the tiny debug dataset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # 2. Directories
    # Create a specific directory for this demo execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.OUTPUT_DIR = "./working/demo_execution"

    # Ensure clean slate
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Update dependent paths
    # Since these are defined at class level in config.py, we must update them manually
    # to reflect the new WORKING_DIR
    Config.TRAIN_X_CACHE = os.path.join(Config.WORKING_DIR, "train_x.npy")
    Config.TRAIN_Y_CACHE = os.path.join(Config.WORKING_DIR, "train_y.npy")
    Config.TRAIN_U_OUT_CACHE = os.path.join(Config.WORKING_DIR, "train_u_out.npy")

    Config.VAL_X_CACHE = os.path.join(Config.WORKING_DIR, "val_x.npy")
    Config.VAL_Y_CACHE = os.path.join(Config.WORKING_DIR, "val_y.npy")
    Config.VAL_U_OUT_CACHE = os.path.join(Config.WORKING_DIR, "val_u_out.npy")

    Config.TEST_X_CACHE = os.path.join(Config.WORKING_DIR, "test_x.npy")
    Config.TEST_U_OUT_CACHE = os.path.join(Config.WORKING_DIR, "test_u_out.npy")

    Config.SCALER_CACHE = os.path.join(Config.WORKING_DIR, "scaler_stats.npz")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    print(f"Working directory set to: {Config.WORKING_DIR}")


def verify_data_pipeline():
    """
    Tests the data loading and preprocessing pipeline.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Force re-processing by setting load_cached_data=False
    # Debug=True loads only 100 breaths (100 * 80 = 8000 rows)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    X, y, u_out = batch["X"], batch["y"], batch["u_out"]

    print(f"Batch X shape: {X.shape}")
    print(f"Batch y shape: {y.shape}")
    print(f"Batch u_out shape: {u_out.shape}")

    # Assertions
    # Shape: (Batch_Size, Seq_Len, Num_Features)
    assert X.ndim == 3, "Input X should be 3-dimensional"
    assert X.shape[1] == Config.SEQ_LEN, f"Sequence length should be {Config.SEQ_LEN}"
    assert (
        X.shape[2] == Config.NUM_FEATURES
    ), f"Feature count should be {Config.NUM_FEATURES}"

    # Target Shape: (Batch_Size, Seq_Len, 1)
    assert y.ndim == 3
    assert y.shape[1] == Config.SEQ_LEN
    assert y.shape[2] == 1

    # u_out Shape: (Batch_Size, Seq_Len, 1)
    assert u_out.ndim == 3

    # Check cache files creation
    assert os.path.exists(Config.TRAIN_X_CACHE), "Train cache file not created"

    print("Data pipeline verification passed.")
    return train_loader, val_loader, test_loader


def verify_model_architecture(device):
    """
    Tests the model instantiation and forward pass.
    """
    print("\n=== Verifying Model Architecture ===")

    model = RPCNet().to(device)
    model.eval()

    # Create dummy input: (Batch=2, Seq_Len=80, Features=14)
    dummy_input = torch.randn(2, Config.SEQ_LEN, Config.NUM_FEATURES).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")

    # Assertions
    assert output.shape == (2, Config.SEQ_LEN, 1), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    print("Model architecture verification passed.")


def verify_training_execution(train_loader, val_loader, test_loader):
    """
    Runs the full training loop and verifies artifact generation.
    """
    print("\n=== Executing Training Loop (Demo) ===")

    # Run the training utility provided in the library
    # This handles training, validation, and submission generation
    run_training(train_loader, val_loader, test_loader)

    # Verify Model Checkpoint
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not saved."
    print(f"Checkpoint found at {Config.MODEL_CHECKPOINT}")

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    # In debug mode, we loaded 100 breaths for test set -> 8000 rows
    expected_rows = 100 * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission missing required columns"
    assert sub_df["pressure"].isnull().sum() == 0, "Submission contains NaN predictions"

    print("Training execution and submission verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Setup
    setup_demo_config()

    # 2. Verify Data
    train_loader, val_loader, test_loader = verify_data_pipeline()

    # 3. Verify Model
    verify_model_architecture(device)

    # 4. Run Training & Inference
    verify_training_execution(train_loader, val_loader, test_loader)

    print("\nAll demonstrations completed successfully.")
