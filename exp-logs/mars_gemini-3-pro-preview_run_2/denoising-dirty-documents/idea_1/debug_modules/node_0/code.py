import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.model import FlatCNN
from library.dataset import load_processed_data, DenoisingDataset
from library.train import run_training
from library.inference import generate_submission


def main():
    print("=== Starting Image Denoising Task Demonstration ===")

    # --- 1. Configure for Fast Demonstration ---
    print("\n[1] Configuring environment for fast execution...")
    # Modify Config attributes to run a quick debug pass
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 images
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug run

    # Ensure reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, SUBSET=10")

    # --- 2. Verify Utility Functions ---
    print("\n[2] Verifying utility functions...")
    # Test RMSE calculation
    preds = np.array([1.0, 0.5, 0.0])
    targets = np.array([1.0, 0.5, 0.0])
    rmse = calculate_rmse(preds, targets)
    assert rmse == 0.0, f"Expected RMSE 0.0, got {rmse}"

    preds = np.array([1.0, 1.0])
    targets = np.array([0.0, 0.0])
    rmse = calculate_rmse(preds, targets)
    assert np.isclose(rmse, 1.0), f"Expected RMSE 1.0, got {rmse}"
    print("calculate_rmse logic verified.")

    # --- 3. Verify Data Loading & Dataset ---
    print("\n[3] Verifying Data Loading and Dataset...")

    # Load processed data (this triggers caching logic)
    train_data = load_processed_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=False
    )

    # Assertions on loaded data
    assert (
        len(train_data) <= Config.DEBUG_SUBSET_SIZE
    ), "Data loading did not respect DEBUG_SUBSET_SIZE"
    assert "noisy" in train_data[0], "Loaded data missing 'noisy' key"
    assert "clean" in train_data[0], "Loaded data missing 'clean' key"
    print(f"Loaded {len(train_data)} training samples successfully.")

    # Instantiate Dataset
    train_dataset = DenoisingDataset(
        train_data, mode="train", patch_size=Config.PATCH_SIZE
    )

    # Test __getitem__
    noisy_t, clean_t = train_dataset[0]

    # Check types
    assert isinstance(noisy_t, torch.Tensor), "Dataset output is not a Tensor"
    assert isinstance(clean_t, torch.Tensor), "Dataset output is not a Tensor"

    # Check shapes: (1, PATCH_SIZE, PATCH_SIZE)
    expected_shape = (1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    assert (
        noisy_t.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {noisy_t.shape}"
    assert (
        clean_t.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {clean_t.shape}"
    print("DenoisingDataset logic verified (Shape and Type).")

    # --- 4. Verify Model Architecture ---
    print("\n[4] Verifying Model Architecture...")
    device = torch.device("cpu")  # Use CPU for simple logic check
    model = FlatCNN().to(device)

    # Create dummy input batch: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(device)

    # Forward pass
    output = model(dummy_input)

    # Assert output shape matches input shape (Flat-CNN property)
    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. In: {dummy_input.shape}, Out: {output.shape}"

    # Assert output range (Sigmoid should be 0-1)
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output values out of range [0, 1]"
    print("FlatCNN architecture verified.")

    # --- 5. Run Training Loop ---
    print("\n[5] Running Training Loop...")
    # This function handles the training loop, validation, and model saving
    # We pass explicit args to ensure our debug config is respected where applicable
    run_training(
        load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
    )

    # Verify model file was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print("Training complete. Model saved.")

    # --- 6. Run Inference and Submission Generation ---
    print("\n[6] Running Inference and Submission Generation...")
    # This function loads the saved model and generates the CSV
    generate_submission(load_cached_data=True, batch_size=1)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate Submission CSV format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert list(df_sub.columns) == ["id", "value"], f"Invalid columns: {df_sub.columns}"

    # Check content types
    assert df_sub["id"].dtype == object, "ID column should be object/string"

    # Check if values are within valid range (0-1)
    # Note: Using a tolerance for float comparison or strict bounds
    assert df_sub["value"].min() >= 0, "Found negative pixel values in submission"
    assert df_sub["value"].max() <= 1, "Found pixel values > 1 in submission"

    print(f"Submission generated successfully with {len(df_sub)} rows.")
    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
