import os
import torch
import pandas as pd
import numpy as np
import shutil

# 1. Configure Environment and Hyperparameters for Demo
from library.config import Config

# Define a separate working directory for this demonstration to avoid conflicts
DEMO_DIR = "./working/demo_execution"
os.makedirs(DEMO_DIR, exist_ok=True)

# Override Config paths
Config.WORKING_DIR = DEMO_DIR
Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

# Important: Manually update derived paths since they were evaluated at import time
Config.PROCESSED_DATA_PATH = os.path.join(Config.CACHE_DIR, "processed_data.npz")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Create necessary subdirectories
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Override Hyperparameters for Speed
Config.DEBUG = True  # Use a small subset of data
Config.DEBUG_SUBSET_SIZE = 50  # Number of samples for debug
Config.NUM_EPOCHS = 1  # Train for only 1 epoch
Config.NUM_FOLDS = 2  # Use only 2 folds instead of 5
Config.BATCH_SIZE = 8  # Small batch size
Config.PATIENCE = 1  # minimal patience

# 2. Imports from Library
from library.utils import set_seed
from library.data_loader import process_and_cache_data
from library.model import MLCWNet
from library.train_eval import train_model, generate_submission

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    print("Configuration and Seeding complete.")

    # ==========================================
    # 3. Data Loading & Processing
    # ==========================================
    print("\n[Step 1] Testing Data Processing and Loading...")

    # Run data processing
    # This will load from json, process, and save to the new Config.PROCESSED_DATA_PATH
    X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test = (
        process_and_cache_data(load_cached_data=False)
    )

    # Assertions to verify data integrity
    print(f"  Training Data Shape: {X_train.shape}")
    print(f"  Test Data Shape: {X_test.shape}")

    # Expected shape: (N, 3, 75, 75)
    assert len(X_train.shape) == 4, "X_train should be 4D"
    assert X_train.shape[1] == 3, "X_train should have 3 channels"
    assert (
        X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Image dimensions should be 75x75"
    assert len(y_train) == len(X_train), "Mismatch between X_train and y_train length"
    assert len(angles_train) == len(
        X_train
    ), "Mismatch between X_train and angles_train length"

    print("  Data Loading verification passed.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[Step 2] Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple architecture check
    model = MLCWNet().to(device)

    # Create dummy input
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_angle = torch.randn(batch_size).to(device)

    # Forward pass
    output = model(dummy_img, dummy_angle)

    print(f"  Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("  Model architecture verification passed.")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n[Step 3] Testing Training Loop (Debug Mode)...")

    # This will use the overridden Config (DEBUG=True, NUM_FOLDS=2, NUM_EPOCHS=1)
    # It saves models to Config.WORKING_DIR
    train_model(num_epochs=Config.NUM_EPOCHS)

    # Verify model files were created
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"mlcw_net_fold_{fold}.pth")
        assert os.path.exists(
            model_path
        ), f"Model file for fold {fold} not found at {model_path}"
        print(f"  Verified model file: {os.path.basename(model_path)}")

    print("  Training loop verification passed.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[Step 4] Testing Inference and Submission Generation...")

    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Shape: {df_sub.shape}")
    print(f"  Submission Columns: {df_sub.columns.tolist()}")

    # Check against expected test set size (321 based on provided info/metadata)
    # Note: process_and_cache_data returns the full test set even in debug mode usually,
    # but get_test_loader might not subset unless we explicitly told it to.
    # The provided get_test_loader does NOT subset based on DEBUG flag, only get_kfold_loaders does.
    # So we expect full test set size.
    assert len(df_sub) == 321, f"Expected 321 rows in submission, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Missing required columns"

    # Check value range
    preds = df_sub["is_iceberg"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("  Inference verification passed.")

    print("\nAll demonstrations completed successfully.")
