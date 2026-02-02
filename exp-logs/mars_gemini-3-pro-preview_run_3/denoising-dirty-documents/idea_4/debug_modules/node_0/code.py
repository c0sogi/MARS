import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import cv2

# Import from the provided library
from library.config import Config
from library.model import DnCNN
from library.data import prepare_data
from library.train import Trainer
from library.predict import InferenceEngine
from library.utils import calculate_rmse


def run_demo():
    print("=== Starting Denoising Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Define a temporary working directory for this demo
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Modify Config attributes to ensure the run finishes quickly
    Config.WORKING_DIR = demo_working_dir
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.MAX_TRAIN_IMAGES = 10  # Limit training data
    Config.MAX_VAL_IMAGES = 5  # Limit validation data
    Config.DEPTH = 5  # Reduce model depth for speed (default is 17)

    # Update paths to point to the demo directory
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_patches.npy")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_patches.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure reproducibility
    Config.set_seed(42)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Model Depth: {Config.DEPTH}")
    print(f"    Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Model Architecture Validation
    # -------------------------------------------------------------------------
    print("\n[2] Validating Model Architecture...")

    # Instantiate the model manually to check forward pass
    model = DnCNN(depth=Config.DEPTH, n_channels=16, image_channels=1)
    model.to(Config.DEVICE)
    model.eval()

    # Create a dummy input tensor: (Batch=2, Channels=1, Height=64, Width=64)
    dummy_input = torch.randn(2, 1, 64, 64).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    # Assert output shape matches input shape (Denoising preserves dimensions)
    assert (
        output.shape == dummy_input.shape
    ), f"Shape mismatch! Input: {dummy_input.shape}, Output: {output.shape}"

    print("    Model forward pass successful. Output shape matches input.")

    # -------------------------------------------------------------------------
    # 3. Data Preparation Validation
    # -------------------------------------------------------------------------
    print("\n[3] Validating Data Preparation...")

    # Explicitly call prepare_data to generate the cache and verify structure
    # load_cached_data=False forces re-computation
    train_data = prepare_data(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH, load_cached_data=False
    )

    # Verify the returned data structure
    assert isinstance(train_data, np.ndarray), "prepare_data must return a numpy array"

    # If data was generated (it should be, given the input files), check dimensions
    if len(train_data) > 0:
        # Expected shape: (N_patches, 2, Patch_H, Patch_W)
        assert train_data.ndim == 4, f"Expected 4D array, got {train_data.ndim}D"
        assert train_data.shape[1] == 2, "Second dimension must be 2 (noisy, clean)"
        assert train_data.shape[2] == Config.PATCH_SIZE, "Patch height mismatch"

        # Check normalization (values should be roughly 0-1)
        # Note: Input images might be all black or white, but range shouldn't exceed [0, 1] significantly
        assert train_data.max() <= 1.0 + 1e-6, "Data not normalized to [0, 1]"
        assert train_data.min() >= 0.0 - 1e-6, "Data contains negative values"

        print(f"    Data loaded successfully. Shape: {train_data.shape}")
    else:
        print("    Warning: No patches extracted. Check input images or patch size.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop...")

    trainer = Trainer()
    # Train using the data prepared in step 3 (loaded from cache)
    trainer.train(load_cached_data=True)

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"

    print("    Training complete. Model saved.")

    # -------------------------------------------------------------------------
    # 5. Inference Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference on Test Set...")

    inference_engine = InferenceEngine()
    inference_engine.run()

    # Verify submission file creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    print("    Inference complete. Submission file generated.")

    # -------------------------------------------------------------------------
    # 6. Submission & Utility Validation
    # -------------------------------------------------------------------------
    print("\n[6] Validating Submission and Utilities...")

    # Validate Submission File Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check required columns
    assert list(df_sub.columns) == ["id", "value"], f"Invalid columns: {df_sub.columns}"

    if not df_sub.empty:
        # Check value constraints
        assert df_sub["value"].min() >= 0, "Found pixel values < 0"
        assert df_sub["value"].max() <= 1, "Found pixel values > 1"

        # Check ID format (image_row_col)
        sample_id = str(df_sub.iloc[0]["id"])
        assert (
            len(sample_id.split("_")) >= 3
        ), f"Invalid ID format: {sample_id}. Expected format like '110_1_1'"

    print("    Submission format is valid.")

    # Validate RMSE Calculation
    true_vals = np.array([0.0, 1.0, 0.5])
    pred_vals = np.array([0.0, 1.0, 0.5])
    rmse_zero = calculate_rmse(true_vals, pred_vals)
    assert rmse_zero == 0.0, f"RMSE should be 0 for identical arrays, got {rmse_zero}"

    pred_vals_off = np.array([0.1, 1.1, 0.6])  # All off by 0.1
    rmse_check = calculate_rmse(true_vals, pred_vals_off)
    assert np.isclose(
        rmse_check, 0.1, atol=1e-6
    ), f"RMSE calculation incorrect. Got {rmse_check}"

    print("    RMSE utility verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Run the full demonstration
    run_demo()
