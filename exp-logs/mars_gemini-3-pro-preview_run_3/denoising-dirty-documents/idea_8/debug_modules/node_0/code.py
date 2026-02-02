import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.data_loader import prepare_data, get_dataloaders
from library.model import ResDnCNN
from library.train import train_model
from library.inference import predict_and_save

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("=== Starting Demonstration Script ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("--- Step 1: Configuring Environment ---")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to point to the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_PATCHES_PATH = os.path.join(DEMO_DIR, "train_patches.npy")
    Config.TRAIN_TARGETS_PATH = os.path.join(DEMO_DIR, "train_targets.npy")
    Config.VAL_PATCHES_PATH = os.path.join(DEMO_DIR, "val_patches.npy")
    Config.VAL_TARGETS_PATH = os.path.join(DEMO_DIR, "val_targets.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Override Hyperparameters for fast execution
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.STRIDE = 200  # Large stride -> fewer patches -> faster processing
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_RES_BLOCKS = 2  # Shallower network for faster forward/backward pass
    Config.TTA_ENABLED = False  # Disable Test-Time Augmentation for speed
    Config.NUM_WORKERS = 2  # Reduce workers for simple demo

    # Create a mini test set for inference demonstration
    # We read the original test csv, take top 3 rows, and save to a new file
    full_test_df = pd.read_csv(Config.TEST_CSV)
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")
    full_test_df.head(3).to_csv(mini_test_path, index=False)
    Config.TEST_CSV = mini_test_path  # Point Config to the mini test set

    Config.print_config()
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("--- Step 2: Verifying Utilities ---")

    # Test RMSE Calculation
    y_true = np.array([0.0, 1.0, 0.5])
    y_pred = np.array([0.0, 1.0, 0.5])
    rmse_val = calculate_rmse(y_true, y_pred)
    assert np.isclose(
        rmse_val, 0.0
    ), f"RMSE should be 0 for identical arrays, got {rmse_val}"

    y_pred_off = np.array([1.0, 0.0, 1.5])  # Off by 1.0 each
    rmse_val_off = calculate_rmse(y_true, y_pred_off)
    assert np.isclose(rmse_val_off, 1.0), f"RMSE should be 1.0, got {rmse_val_off}"

    print("Utility checks passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading and Preparation
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Data Preparation ---")

    # Force data preparation (load_cached_data=False) to use the new STRIDE
    print("Extracting patches with high stride for demo...")
    (train_x, train_y), (val_x, val_y) = prepare_data(load_cached_data=False)

    # Assertions to verify data integrity
    assert train_x.ndim == 3, "Train patches should be 3D (N, H, W)"
    assert train_x.shape == train_y.shape, "Input and Target shapes must match"
    assert (
        train_x.shape[1] == Config.PATCH_SIZE
    ), f"Height should be {Config.PATCH_SIZE}"
    assert train_x.shape[2] == Config.PATCH_SIZE, f"Width should be {Config.PATCH_SIZE}"
    assert len(train_x) > 0, "No training patches were extracted."

    print(
        f"Extracted {len(train_x)} training patches and {len(val_x)} validation patches."
    )

    # Verify DataLoader
    train_loader, val_loader = get_dataloaders(load_cached_data=True)
    sample_batch_x, sample_batch_y = next(iter(train_loader))

    # Shape check: [Batch, Channel, Height, Width]
    expected_shape = (Config.BATCH_SIZE, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    # Note: The last batch might be smaller if drop_last=False (default),
    # but with 16 BS and >16 samples, the first batch should be full size.
    if len(train_x) >= Config.BATCH_SIZE:
        assert (
            sample_batch_x.shape == expected_shape
        ), f"Batch shape mismatch. Expected {expected_shape}, got {sample_batch_x.shape}"

    print("DataLoaders initialized successfully.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Model Verification ---")

    device = torch.device(Config.DEVICE)
    model = ResDnCNN().to(device)

    # Run a dummy forward pass
    dummy_input = sample_batch_x.to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert (
        dummy_output.shape == dummy_input.shape
    ), f"Model output shape {dummy_output.shape} does not match input {dummy_input.shape}"

    print("Model instantiated and forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Running Training Loop ---")

    # This calls the library function which uses the Config we modified
    train_model()

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint was not created at {Config.MODEL_SAVE_PATH}"

    print("Training complete and model saved.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Running Inference ---")

    # Run inference using the mini test set configured earlier
    predict_and_save()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission file missing required columns 'id' and 'value'"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check ID format (e.g., '110_1_1')
    sample_id = df_sub.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) >= 3, f"ID format incorrect: {sample_id}"

    print(f"Submission generated at {Config.SUBMISSION_PATH} with {len(df_sub)} rows.")
    print("\n=== Demonstration Completed Successfully ===")
