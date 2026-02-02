import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def main():
    # --------------------------------------------------------------------------
    # Setup & Configuration Override
    # --------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Manufacturing State Prediction Demo ===")

    # Override default configuration for a fast demonstration
    print("\n[Setup] Overriding configuration for fast execution...")

    # Use a separate cache directory for this demo to avoid conflicts
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Enable Debug mode to use a small subset of data
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 2000  # Use 2000 samples for train/val/test

    # Training hyperparameters for demo
    config.EPOCHS = 2
    config.BATCH_SIZE = 128
    config.NUM_WORKERS = 2

    # Update paths to point to the demo working directory
    config.MODEL_SAVE_PATH = os.path.join(config.CACHE_DIR, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.CACHE_DIR, "submission.csv")

    # Ensure reproducibility
    utils.seed_everything(config.RANDOM_STATE)

    print(f"Debug Mode: {config.DEBUG}")
    print(f"Subset Size: {config.DEBUG_SUBSET_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Device: {config.DEVICE}")

    # --------------------------------------------------------------------------
    # 1. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1] Verifying Data Loading Pipeline...")

    # Generate dataloaders (this will trigger data preprocessing on the first run)
    # We force load_cached_data=False initially to prove preprocessing works
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=False, debug=config.DEBUG
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    cont_features = batch["continuous"]
    cat_features = batch["categorical"]
    targets = batch["target"]
    ids = batch["id"]

    # Assertions to verify data shapes and types
    assert cont_features.shape == (
        config.BATCH_SIZE,
        config.NUM_CONTINUOUS_FEATURES,
    ), f"Continuous features shape mismatch. Expected ({config.BATCH_SIZE}, {config.NUM_CONTINUOUS_FEATURES}), got {cont_features.shape}"

    assert cat_features.shape == (
        config.BATCH_SIZE,
        config.SEQUENCE_LENGTH,
    ), f"Categorical features shape mismatch. Expected ({config.BATCH_SIZE}, {config.SEQUENCE_LENGTH}), got {cat_features.shape}"

    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Target shape mismatch. Expected ({config.BATCH_SIZE},), got {targets.shape}"

    assert ids.shape == (
        config.BATCH_SIZE,
    ), f"ID shape mismatch. Expected ({config.BATCH_SIZE},), got {ids.shape}"

    print("Data Loading Verified: Batch shapes are correct.")

    # --------------------------------------------------------------------------
    # 2. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    # Instantiate the model
    net = model.ManufacturingNet()
    net.to(config.DEVICE)
    net.eval()

    # Run a dummy forward pass
    with torch.no_grad():
        # Move data to device
        cont_dev = cont_features.to(config.DEVICE)
        cat_dev = cat_features.to(config.DEVICE)

        # Forward pass
        logits = net(cont_dev, cat_dev)

    # Verify output shape
    # The model uses Multi-Sample Dropout with MSD_NUM_HEADS (default 5)
    expected_shape = (config.BATCH_SIZE, config.MSD_NUM_HEADS)
    assert (
        logits.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {logits.shape}"

    print(f"Model Architecture Verified: Output shape is {logits.shape}.")

    # --------------------------------------------------------------------------
    # 3. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 3] Running Training Loop...")

    # Run training using the library function
    # We use load_cached_data=True to utilize the data processed in Step 1
    best_auc = train.run_training(
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # Verify that a model file was saved
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), f"Model file was not saved at {config.MODEL_SAVE_PATH}"

    print(f"Training Loop Completed. Best AUC: {best_auc:.4f}")

    # --------------------------------------------------------------------------
    # 4. Inference and Submission Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Generating Submission...")

    # Generate submission using the trained model
    train.generate_submission(batch_size=config.BATCH_SIZE, load_cached_data=True)

    # Verify the submission file
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Verify columns
    expected_cols = ["id", "target"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Verify row count
    # In debug mode, the test set is also subsampled to DEBUG_SUBSET_SIZE
    assert (
        len(df_sub) == config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    # Verify value range (probabilities should be between 0 and 1)
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Prediction values are out of probability range [0, 1]"

    print(
        f"Submission Verified: File saved at {config.SUBMISSION_PATH} with {len(df_sub)} rows."
    )
    print("\n=== Demo Execution Successfully Completed ===")


if __name__ == "__main__":
    main()
