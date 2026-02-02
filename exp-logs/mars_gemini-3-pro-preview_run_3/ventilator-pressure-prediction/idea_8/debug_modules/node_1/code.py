import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, compute_mae
from library.data import DataProcessor, VentilatorDataset
from library.model import PSANet
from library.train import Trainer


def main():
    print("=== Starting Demonstration of Ventilator Pressure Prediction Library ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    print("1. Setting up Demo Configuration...")

    class DemoConfig(Config):
        """
        Configuration overrides for a fast demonstration run.
        """

        # Enable Debug mode to use a tiny subset of data
        DEBUG = True
        DEBUG_SAMPLES = 200  # Use 200 breaths for speed

        # Reduce training duration
        EPOCHS = 2
        BATCH_SIZE = 32

        # Set specific directories for this demo to avoid overwriting real work
        WORKING_DIR = "./working/demo_execution"
        SUBMISSION_DIR = "./working/demo_execution"
        SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

        # Re-define cache paths to point to the demo working directory
        CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_x.npy")
        CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
        CACHE_TRAIN_UOUT = os.path.join(WORKING_DIR, "train_u_out.npy")
        CACHE_VAL_X = os.path.join(WORKING_DIR, "val_x.npy")
        CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
        CACHE_VAL_UOUT = os.path.join(WORKING_DIR, "val_u_out.npy")
        CACHE_TEST_X = os.path.join(WORKING_DIR, "test_x.npy")
        CACHE_TEST_UOUT = os.path.join(WORKING_DIR, "test_u_out.npy")

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(DemoConfig.WORKING_DIR):
        shutil.rmtree(DemoConfig.WORKING_DIR)
    os.makedirs(DemoConfig.WORKING_DIR, exist_ok=True)

    seed_everything(DemoConfig.SEED)
    print("   Configuration ready. DEBUG mode enabled.\n")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("2. Verifying Utility Functions (compute_mae)...")

    # Create dummy data:
    # Preds: [10, 10], Targets: [12, 10], u_out: [0, 1]
    # Index 0: u_out=0 (Inspiratory) -> |10 - 12| = 2.0
    # Index 1: u_out=1 (Expiratory)  -> Ignored
    # Expected MAE: 2.0
    dummy_preds = torch.tensor([10.0, 10.0])
    dummy_targets = torch.tensor([12.0, 10.0])
    dummy_u_out = torch.tensor([0.0, 1.0])

    mae = compute_mae(dummy_preds, dummy_targets, dummy_u_out)
    print(f"   Calculated MAE: {mae}")

    assert mae == 2.0, f"Expected MAE 2.0, got {mae}"
    print("   Assertion Passed: compute_mae logic is correct.\n")

    # --------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("3. Verifying PSANet Model Architecture...")

    model = PSANet(DemoConfig)
    # Input shape: (Batch, Seq_Len, Features)
    dummy_input = torch.randn(4, DemoConfig.SEQ_LEN, DemoConfig.INPUT_DIM)

    print(f"   Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")

    # Expected output shape: (Batch, Seq_Len)
    expected_shape = (4, DemoConfig.SEQ_LEN)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"
    print("   Assertion Passed: Model forward pass successful.\n")

    # --------------------------------------------------------------------------
    # 4. Verify Data Processing
    # --------------------------------------------------------------------------
    print("4. Verifying DataProcessor...")

    processor = DataProcessor(DemoConfig)

    # Run data preparation (Load -> Feature Engineer -> Scale -> Reshape)
    # This will use the metadata files and save numpy arrays to WORKING_DIR
    (train_data, val_data, test_data) = processor.prepare_data(load_cached_data=False)

    train_x, train_y, train_u_out = train_data
    val_x, val_y, val_u_out = val_data
    test_x, _, test_u_out = test_data

    print(f"   Train X shape: {train_x.shape}")
    print(f"   Val X shape:   {val_x.shape}")
    print(f"   Test X shape:  {test_x.shape}")

    # Assertions
    # 1. Check feature dimension
    assert (
        train_x.shape[2] == DemoConfig.INPUT_DIM
    ), "Incorrect feature dimension in Train X"
    # 2. Check sequence length
    assert (
        train_x.shape[1] == DemoConfig.SEQ_LEN
    ), "Incorrect sequence length in Train X"
    # 3. Check target alignment
    assert (
        train_x.shape[0] == train_y.shape[0]
    ), "Mismatch between Train X and Train Y samples"
    # 4. Check u_out alignment
    assert (
        train_x.shape[0] == train_u_out.shape[0]
    ), "Mismatch between Train X and Train u_out"

    print("   Assertion Passed: Data shapes are correct.\n")

    # --------------------------------------------------------------------------
    # 5. Verify Dataset Class
    # --------------------------------------------------------------------------
    print("5. Verifying VentilatorDataset...")

    dataset = VentilatorDataset(train_x, train_y, train_u_out)
    sample = dataset[0]

    print(f"   Sample keys: {sample.keys()}")
    assert (
        "x" in sample and "y" in sample and "u_out" in sample
    ), "Missing keys in dataset sample"
    assert isinstance(sample["x"], torch.Tensor), "Dataset should return tensors"

    print("   Assertion Passed: Dataset yields correct format.\n")

    # --------------------------------------------------------------------------
    # 6. Run Full Training Pipeline
    # --------------------------------------------------------------------------
    print("6. Running Full Training Pipeline (Trainer)...")

    trainer = Trainer(DemoConfig)

    # This runs: Data Prep (cached) -> Train Loop -> Validation -> Prediction -> Submission
    trainer.run()

    print("\n   Pipeline execution finished.")

    # --------------------------------------------------------------------------
    # 7. Verify Submission Output
    # --------------------------------------------------------------------------
    print("7. Verifying Submission File...")

    if not os.path.exists(DemoConfig.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {DemoConfig.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(DemoConfig.SUBMISSION_PATH)
    print(f"   Submission head:\n{submission_df.head()}")
    print(f"   Submission shape: {submission_df.shape}")

    # Check columns
    assert list(submission_df.columns) == [
        "id",
        "pressure",
    ], "Submission columns mismatch"

    # Check length
    # In DEBUG mode, we selected DEBUG_SAMPLES (200) breaths.
    # Each breath has SEQ_LEN (80) time steps.
    # Total rows should be 200 * 80 = 16000.
    expected_rows = DemoConfig.DEBUG_SAMPLES * DemoConfig.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(submission_df)}"

    print("   Assertion Passed: Submission file format and length are correct.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
