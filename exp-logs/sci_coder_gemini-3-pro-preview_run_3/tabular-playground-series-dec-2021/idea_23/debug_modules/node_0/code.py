import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.model import DeepParallelDCNResNet
from library.data_utils import get_datasets
from library.train import run_training, Trainer
from library.inference import generate_predictions


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("=== Step 1: Configuration Setup ===")
    set_seed(42)

    # Modify Config for a fast demonstration (Debug Mode)
    # We override class attributes directly to affect all downstream modules
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small subset for speed
    Config.EPOCHS = 1  # Only 1 epoch for demonstration
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory is clean for this run to avoid cache conflicts
    # (Optional but good practice for a standalone demo)
    if os.path.exists(Config.CACHE_DIR):
        print(f"Cleaning cache directory: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)

    Config.create_directories()
    print("Configuration updated for debug run.")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing Verification
    # -------------------------------------------------------------------------
    print("\n=== Step 2: Data Loading & Processing Verification ===")

    # Load datasets using the utility function with caching disabled to force processing
    train_dataset, val_dataset, test_dataset, test_ids, classes = get_datasets(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Verify Dataset Types
    print(f"Train Dataset Type: {type(train_dataset)}")
    assert isinstance(
        train_dataset, torch.utils.data.TensorDataset
    ), "Train dataset is not a TensorDataset"

    # Verify Dataset Sizes (should match DEBUG_SAMPLES)
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Val Samples:   {len(val_dataset)}")
    print(f"Test Samples:  {len(test_dataset)}")

    assert (
        len(train_dataset) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} train samples"
    assert (
        len(val_dataset) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} val samples"
    assert (
        len(test_dataset) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} test samples"

    # Verify Feature Dimensions
    # train_dataset[0] is (features, target)
    input_features = train_dataset[0][0]
    input_dim = input_features.shape[0]
    num_classes = len(classes)

    print(f"Input Feature Dimension: {input_dim}")
    print(f"Number of Classes: {num_classes}")

    assert input_dim > 0, "Input dimension must be positive"
    assert num_classes > 1, "Number of classes must be > 1"

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Model Architecture Verification ===")

    # Instantiate the model
    device = torch.device("cpu")  # Use CPU for simple logic verification
    model = DeepParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=2,  # Reduce blocks for demo speed
        dropout_rate=0.1,
    ).to(device)

    print("Model instantiated successfully.")

    # Create a dummy batch input [Batch_Size, Input_Dim]
    dummy_batch_size = 4
    dummy_input = torch.randn(dummy_batch_size, input_dim).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        dummy_batch_size,
        num_classes,
    ), f"Expected output shape {(dummy_batch_size, num_classes)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # -------------------------------------------------------------------------
    # 4. Full Training Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Running Full Training Pipeline ===")

    # We use the provided `run_training` function which handles the Trainer loop
    # We pass load_cached_data=True because we generated the cache in Step 2.
    # debug=True is passed to ensure consistency, though Config.DEBUG is already set.

    run_training(load_cached_data=True, debug=True)

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print(f"Training complete. Best model saved to {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Verification
    # -------------------------------------------------------------------------
    print("\n=== Step 5: Inference and Submission Verification ===")

    # Run inference using the saved model
    # We force device to match what's available (likely CPU or GPU from Config)
    generate_predictions(
        load_cached_data=True, batch_size=Config.BATCH_SIZE, device_name=Config.DEVICE
    )

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Assertions on Submission
    assert Config.ID_COL in df_sub.columns, f"Missing ID column {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Missing Target column {Config.TARGET_COL}"
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} predictions, got {len(df_sub)}"

    # Check for nulls
    assert df_sub.isnull().sum().sum() == 0, "Submission contains null values"

    # Check if predictions are valid integers (classes)
    # Since we mapped back to original classes, they should be integers like 1, 2, 3...
    assert pd.api.types.is_integer_dtype(
        df_sub[Config.TARGET_COL]
    ), "Target column should be integer type"

    print("\n=== Demo Completed Successfully ===")
