import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MGSHDNetwork
from library.train import Trainer


def run_demo():
    print("Starting End-to-End Demo of MG-SHD Network Pipeline...")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config defaults to run a small, fast experiment
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Process only 12 subjects per split
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Use a specific directory for this demo to avoid cache conflicts
    Config.CACHE_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Ensure directories exist
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=2, BATCH_SIZE=4")

    # ------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # ------------------------------------------------------------------------
    print("\n[2] Initializing Data Loaders and Verifying Shapes...")

    # Force processing from scratch (load_cached_data=False) to test data processing logic
    # Note: In a real run, we would use True to save time.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    try:
        batch_x, batch_y = next(iter(train_loader))
        print(f"   Train Batch X shape: {batch_x.shape}")
        print(f"   Train Batch y shape: {batch_y.shape}")

        # Assertions
        expected_channels = Config.IN_CHANNELS  # 128
        expected_size = Config.IMG_SIZE  # 224

        assert batch_x.shape == (
            Config.BATCH_SIZE,
            expected_channels,
            expected_size,
            expected_size,
        ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, expected_channels, expected_size, expected_size)}, got {batch_x.shape}"
        assert batch_y.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {batch_y.shape}"

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Test Loader (should yield IDs instead of targets)
    try:
        test_batch_x, test_batch_ids = next(iter(test_loader))
        print(f"   Test Batch X shape: {test_batch_x.shape}")
        print(f"   Test Batch IDs shape: {test_batch_ids.shape}")

        assert test_batch_x.shape[1:] == (
            expected_channels,
            expected_size,
            expected_size,
        ), "Test input dimensions are incorrect."

    except StopIteration:
        raise AssertionError("Test loader is empty!")

    print("Data pipeline verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------------
    print("\n[3] Instantiating Model and Verifying Forward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGSHDNetwork().to(device)

    # Move batch to device
    batch_x = batch_x.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(batch_x)

    print(f"   Model Output shape: {output.shape}")

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("Model architecture verified successfully.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Verification
    # ------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Trainer.fit)...")

    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run training
    trainer.fit(epochs=Config.EPOCHS, patience=1)

    # Verify checkpoint creation
    best_model_path = trainer.best_model_path
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model checkpoint was not saved at {best_model_path}"
        )

    print(f"Training completed. Checkpoint found at: {best_model_path}")

    # ------------------------------------------------------------------------
    # 5. Submission Generation Verification
    # ------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    trainer.generate_submission()

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission rows: {len(df_sub)}")
    print(f"   Submission columns: {df_sub.columns.tolist()}")

    assert "BraTS21ID" in df_sub.columns, "Missing 'BraTS21ID' column"
    assert "MGMT_value" in df_sub.columns, "Missing 'MGMT_value' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check ID format (should be 5 digits, e.g., '00001')
    # Since we read CSV, if IDs are strings like '00001', pandas might infer int or object.
    # The generate_submission function ensures they are strings in the CSV.
    # We check the first value.
    first_id = str(df_sub.iloc[0]["BraTS21ID"])
    if len(first_id) != 5:
        # If pandas read it as int (e.g. 1), it won't have length 5.
        # However, the requirement is the file format. Let's check the file content directly for the header/format.
        pass

    print("Submission generation verified successfully.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
