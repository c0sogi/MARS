import sys
import os
import shutil
import torch
import pandas as pd
import numpy as np

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.data_processor import DataProcessor
from library.dataset import TaxiDataset
from library.model import SpatialResNet
from library.trainer import Trainer
from library.utils import seed_everything


def main():
    # =========================================
    # 1. Configuration & Setup
    # =========================================
    print("[1/6] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20000  # Use only 20k rows for speed
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 256
    Config.NUM_WORKERS = 0  # Use main process for data loading to avoid overhead
    Config.NUM_RES_BLOCKS = 2  # Reduce model depth for faster inference in demo

    # Update paths based on new working dir
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG} (Size: {Config.DEBUG_SAMPLE_SIZE})")

    # =========================================
    # 2. Data Processing
    # =========================================
    print("\n[2/6] Running DataProcessor...")
    processor = DataProcessor()

    # Process data from scratch (load_cached_data=False) to test the pipeline
    train_df, val_df, test_df = processor.process_data(load_cached_data=False)

    # Validate Data Processing
    print("    Verifying processed data shapes...")
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train set size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_df)}"
    assert (
        len(val_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val set size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(val_df)}"

    # Check for expected features
    expected_cols = processor.continuous_cols + processor.categorical_cols
    for col in expected_cols:
        assert col in train_df.columns, f"Missing column {col} in processed data."

    print("    Data processing verified successfully.")

    # =========================================
    # 3. Dataset Verification
    # =========================================
    print("\n[3/6] Verifying TaxiDataset...")
    # Initialize dataset (will load the cached data we just created)
    train_dataset = TaxiDataset(split="train", load_cached_data=True)

    # Fetch a single sample
    sample = train_dataset[0]

    # Verify sample structure
    assert "continuous_features" in sample
    assert "spatial_indices" in sample
    assert "target" in sample

    # Verify Tensor shapes
    # Continuous: 11 features
    assert sample["continuous_features"].shape == (
        11,
    ), f"Expected continuous shape (11,), got {sample['continuous_features'].shape}"
    # Categorical: 9 features
    assert sample["spatial_indices"].shape == (
        9,
    ), f"Expected spatial indices shape (9,), got {sample['spatial_indices'].shape}"

    print("    Dataset structure verified.")

    # =========================================
    # 4. Model Architecture Verification
    # =========================================
    print("\n[4/6] Verifying SpatialResNet Architecture...")
    model = SpatialResNet(
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
        grid_bins=Config.GRID_BINS,
    )

    # Perform a dummy forward pass
    # Add batch dimension (unsqueeze)
    cont_input = sample["continuous_features"].unsqueeze(0)  # Shape: (1, 11)
    cat_input = sample["spatial_indices"].unsqueeze(0)  # Shape: (1, 9)

    model.eval()
    with torch.no_grad():
        output = model(cont_input, cat_input)

    # Verify output shape (Batch Size, 1)
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    print("    Model forward pass successful.")

    # =========================================
    # 5. Training Loop
    # =========================================
    print("\n[5/6] Executing Training Loop...")
    trainer = Trainer()

    # Run training
    # This uses the parameters set in Config (2 epochs, debug subset)
    trainer.fit()

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print("    Training complete and model saved.")

    # =========================================
    # 6. Submission Generation & Validation
    # =========================================
    print("\n[6/6] Generating and Validating Submission...")
    trainer.generate_submission()

    # Verify submission file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV not found."

    # Load submission to check content
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    test_meta_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # 1. Check Row Count
    assert len(submission_df) == len(
        test_meta_df
    ), f"Submission row count mismatch. Expected {len(test_meta_df)}, got {len(submission_df)}"

    # 2. Check Columns
    assert "key" in submission_df.columns, "Column 'key' missing in submission."
    assert (
        "fare_amount" in submission_df.columns
    ), "Column 'fare_amount' missing in submission."

    # 3. Check for NaNs
    assert (
        submission_df["fare_amount"].isnull().sum() == 0
    ), "Found null values in fare_amount predictions."

    # 4. Check Value Constraints (Min Prediction Floor)
    min_pred = submission_df["fare_amount"].min()
    assert (
        min_pred >= Config.MIN_FARE_PREDICTION
    ), f"Found predictions below minimum floor {Config.MIN_FARE_PREDICTION}. Min found: {min_pred}"

    print(f"    Submission valid. Sample:\n{submission_df.head(3)}")
    print("\nSUCCESS: All steps completed and verified.")


if __name__ == "__main__":
    main()
