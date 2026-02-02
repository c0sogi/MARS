import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path to allow library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import get_dataloaders
from library.model import CompositeCNN, set_seeds
from library.train import run_training
from library.predict import predict


def main():
    print("=== Starting Demonstration and Verification Script ===")

    # 1. Setup
    # Set seeds for reproducibility across all operations
    set_seeds(42)
    config = Config()

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Input Directory: {config.INPUT_DIR}")

    # 2. Data Loading Verification
    print("\n--- Step 1: Verifying Data Loading and Processing ---")

    # We set load_cached_data=False to demonstrate the raw data processing logic.
    # This will read from metadata and json files, process them, and save .npy caches.
    print("Initializing DataLoaders (processing raw data)...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch from the training loader
    try:
        imgs, angles, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(
        f"Batch Shapes -> Images: {imgs.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions to verify data integrity
    expected_img_shape = (config.BATCH_SIZE, 3, 75, 75)
    assert (
        imgs.shape == expected_img_shape
    ), f"Expected image shape {expected_img_shape}, got {imgs.shape}"
    assert angles.shape == (
        config.BATCH_SIZE,
    ), f"Expected angle shape {(config.BATCH_SIZE,)}, got {angles.shape}"
    assert labels.shape == (
        config.BATCH_SIZE,
    ), f"Expected label shape {(config.BATCH_SIZE,)}, got {labels.shape}"
    assert imgs.dtype == torch.float32, "Images should be float32 tensor"

    # Verify Test Loader
    test_imgs, test_angles = next(iter(test_loader))
    assert test_imgs.shape == expected_img_shape, "Test image shape mismatch"

    print("Data Loading verified successfully.")

    # 3. Model Architecture Verification
    print("\n--- Step 2: Verifying Model Architecture ---")

    model = CompositeCNN(config)

    # Move batch to CPU (default for this script) or CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    imgs = imgs.to(device)
    angles = angles.to(device)

    # Perform forward pass
    output = model(imgs, angles)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        config.BATCH_SIZE,
        1,
    ), "Model output shape should be (Batch_Size, 1)"
    assert torch.all(output >= 0.0) and torch.all(
        output <= 1.0
    ), "Model outputs must be probabilities [0, 1]"

    print("Model Architecture verified successfully.")

    # 4. Training Pipeline Verification
    print("\n--- Step 3: Verifying Training Pipeline (Integration) ---")

    # We run a very short training cycle:
    # - 1 Epoch
    # - Batch Size 4
    # - Max Samples 20 (to make the epoch extremely fast)
    # - Load cached data (generated in Step 1) to save time

    print("Running training simulation...")
    run_training(
        num_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        load_cached_data=True,
        max_samples=20,
    )

    # Verify artifacts
    if not os.path.exists(config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint was not created at {config.MODEL_CHECKPOINT}"
        )

    if not os.path.exists(config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file was not created at {config.SUBMISSION_FILE}"
        )

    # Verify submission content
    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Generated submission with {len(df_sub)} rows.")
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns mismatch"

    print("Training Pipeline verified successfully.")

    # 5. Prediction Pipeline Verification
    print("\n--- Step 4: Verifying Prediction Pipeline (Standalone) ---")

    # Rename previous submission to ensure we are validating a fresh run
    backup_sub = config.SUBMISSION_FILE + ".bak"
    if os.path.exists(config.SUBMISSION_FILE):
        os.rename(config.SUBMISSION_FILE, backup_sub)

    # Run prediction with a limit
    # This simulates loading the saved model and predicting on a subset of test data
    predict(load_cached_data=True, batch_size=4, max_samples=12)

    # Verify new submission
    if not os.path.exists(config.SUBMISSION_FILE):
        raise FileNotFoundError("Prediction pipeline failed to create submission file.")

    df_pred = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Prediction run generated {len(df_pred)} rows.")

    # We requested max_samples=12. The predict function slices the dataset.
    # Depending on batch size (4), it should process exactly 12 samples.
    assert len(df_pred) == 12, f"Expected 12 predictions, got {len(df_pred)}"

    print("Prediction Pipeline verified successfully.")

    # Cleanup (restore original submission if needed, though not strictly required)
    if os.path.exists(backup_sub):
        os.remove(backup_sub)

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
