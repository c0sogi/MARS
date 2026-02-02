import sys
import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloader
from library.model import TemporalCNN
from library.train import train_model, generate_submission


def main():
    # 1. Setup and Reproducibility
    print("--- Setting up environment ---")
    set_seed(Config.SEED)

    # Ensure the working directory matches where the cached files are located
    # (Based on the prompt description, cached files are in ./working/idea_2)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Verify Data Loading and Dataset
    print("\n--- Verifying Data Loading ---")
    # We load the training dataloader.
    # load_cached_data=True will attempt to read .parquet files from Config.WORKING_DIR
    batch_size = 32
    train_loader = get_dataloader("train", batch_size=batch_size, load_cached_data=True)

    # Fetch a single batch to verify shapes and types
    try:
        X_batch, y_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader is empty!")

    print(f"Batch X shape: {X_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")

    # Assertions
    # Expected Input Shape: (Batch, Channels, Time)
    # Channels = 22 (from Config.FEATURES), Time = 9 (Config.WINDOW_SIZE)
    expected_channels = Config.NUM_FEATURES
    expected_time = Config.WINDOW_SIZE

    assert X_batch.shape == (
        batch_size,
        expected_channels,
        expected_time,
    ), f"Input shape mismatch. Expected ({batch_size}, {expected_channels}, {expected_time}), got {X_batch.shape}"

    # Expected Label Shape: (Batch, 1)
    assert y_batch.shape == (
        batch_size,
        1,
    ), f"Label shape mismatch. Expected ({batch_size}, 1), got {y_batch.shape}"

    print("Data Loading and Shapes Verified.")

    # 3. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = TemporalCNN()

    # Move model to CPU for this quick check (or GPU if available, but consistent with tensor)
    device = torch.device("cpu")
    model.to(device)
    X_batch = X_batch.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(X_batch)

    print(f"Model Output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected ({batch_size}, 1), got {output.shape}"

    # Check range (Sigmoid output should be between 0 and 1)
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output values out of range [0, 1]."

    print("Model Architecture Verified.")

    # 4. Verify Training Pipeline (Debug Mode)
    print("\n--- Verifying Training Loop ---")
    # We use debug=True to run only a few batches per epoch and stop after 1 epoch
    # This validates the optimization step, loss calculation, and validation loop without waiting for full training.

    trained_model, best_threshold = train_model(
        num_epochs=1, batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=True
    )

    print(f"Training Loop Completed. Best Threshold: {best_threshold}")

    # 5. Verify Submission Generation (Debug Mode)
    print("\n--- Verifying Submission Generation ---")
    # Generate submission using the trained model
    # debug=True limits the inference to a few batches
    df_submission = generate_submission(
        trained_model,
        best_threshold,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=True,
    )

    print(f"Submission DataFrame Shape: {df_submission.shape}")
    print(f"Submission Columns: {df_submission.columns.tolist()}")

    # Assertions
    assert (
        "contact_id" in df_submission.columns
    ), "Missing 'contact_id' column in submission."
    assert "contact" in df_submission.columns, "Missing 'contact' column in submission."
    assert (
        df_submission["contact"].isin([0, 1]).all()
    ), "Predictions must be binary (0 or 1)."

    # Verify file creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    print("Submission Generation Verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
