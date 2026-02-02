import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import MultiBandResNetCRNN
from library.trainer import Trainer


def main():
    print("Initializing Demo Execution...")

    # 1. Configuration Overrides for Speed and Demonstration
    # We modify the Config class attributes directly to run a fast demo.

    # Use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set paths for cache and submission
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Enable Debug mode to process only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Only use 100 samples for train/val/test

    # Reduce training parameters for speed
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(
        f"Configuration set: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}"
    )
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n--- Step 1: Data Loading ---")
    # We force `load_cached_data=False` to demonstrate the preprocessing pipeline
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verification: Check DataLoaders
    print("Verifying DataLoaders...")

    # Check Train Loader
    try:
        train_batch, train_labels = next(iter(train_loader))
        print(f"Train Batch Shape: {train_batch.shape}")
        print(f"Train Labels Shape: {train_labels.shape}")

        # Assertions
        assert train_batch.dim() == 4, "Train batch should be 4D (B, C, F, T)"
        assert (
            train_batch.shape[0] == Config.BATCH_SIZE
        ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}"
        assert train_batch.shape[1] == 1, "Input channel should be 1"
        assert train_labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Check Test Loader (returns IDs instead of labels)
    try:
        test_batch, test_ids = next(iter(test_loader))
        print(f"Test Batch Shape: {test_batch.shape}")
        assert len(test_ids) == Config.BATCH_SIZE, "Test ID batch size mismatch"
    except StopIteration:
        raise AssertionError("Test loader is empty!")

    print("Data Loading verified successfully.")

    # 3. Model Initialization
    print("\n--- Step 2: Model Initialization ---")
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = MultiBandResNetCRNN().to(device)

    # Verification: Forward Pass
    print("Verifying Model Forward Pass...")
    with torch.no_grad():
        dummy_input = train_batch.to(device)
        output = model(dummy_input)

        print(f"Model Output Shape: {output.shape}")

        # Assertions
        assert output.shape == (
            Config.BATCH_SIZE,
            1,
        ), "Output shape mismatch. Expected (B, 1)"
        assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model initialized and verified successfully.")

    # 4. Training Loop
    print("\n--- Step 3: Training ---")
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run training
    # This uses the Trainer.fit method which handles loops, validation, and saving
    trainer.fit(epochs=Config.EPOCHS)

    # Verification: Check if model checkpoint exists
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print(f"Training verified. Best model found at {best_model_path}")

    # 5. Inference and Submission
    print("\n--- Step 4: Inference & Submission ---")
    trainer.generate_submission(test_loader)

    # Verification: Check Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_submission)}")
    print(df_submission.head())

    # Assertions on Submission
    assert "clip" in df_submission.columns, "Submission missing 'clip' column"
    assert (
        "probability" in df_submission.columns
    ), "Submission missing 'probability' column"

    # Since we used DEBUG mode with subset size 100, the submission should have 100 rows
    # Note: dataset.py process_and_cache_subset slices the dataframe to DEBUG_SUBSET_SIZE
    assert (
        len(df_submission) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_submission)}"

    # Check probability range
    probs = df_submission["probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
