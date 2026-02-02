import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config, set_seed, HybridResFunnel
from library.dataset import get_dataloaders
from library.model import train_and_predict


def run_demo():
    print("=== Manufacturing Control Task: Library Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    print(">>> [Step 1] Setup and Configuration")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Define a specific working directory for this demo run
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config attributes for the demo to ensure speed and isolation
    # We modify the class attributes directly so they propagate to other modules
    Config.WORKING_DIR = demo_dir
    Config.OUTPUT_SUBMISSION = os.path.join(demo_dir, "submission_demo.csv")
    Config.BATCH_SIZE = 512  # Efficient batch size
    Config.EPOCHS = 1  # Single epoch for demonstration

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Output Submission: {Config.OUTPUT_SUBMISSION}")
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n>>> [Step 2] Verifying Data Pipeline")

    # Load data using the library function
    # This handles metadata reading, feature processing (f_27 tokenization), and caching
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch to verify structure
    try:
        x_cat_batch, x_cont_batch, y_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(
        f"Batch Shapes -> Categorical: {x_cat_batch.shape}, Continuous: {x_cont_batch.shape}, Target: {y_batch.shape}"
    )

    # Assertions to guarantee data integrity
    # Categorical: [Batch, 10] (Sequence length 10)
    assert x_cat_batch.shape == (
        Config.BATCH_SIZE,
        10,
    ), f"Expected categorical shape ({Config.BATCH_SIZE}, 10), got {x_cat_batch.shape}"

    # Continuous: [Batch, 30] (30 features)
    assert x_cont_batch.shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Expected continuous shape ({Config.BATCH_SIZE}, 30), got {x_cont_batch.shape}"

    # Target: [Batch, 1]
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected target shape ({Config.BATCH_SIZE}, 1), got {y_batch.shape}"

    print("Data shapes verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n>>> [Step 3] Verifying Model Architecture")

    # Instantiate the model
    model = HybridResFunnel(Config).to(Config.DEVICE)

    # Move batch to device
    x_cat_dev = x_cat_batch.to(Config.DEVICE)
    x_cont_dev = x_cont_batch.to(Config.DEVICE)

    # Perform a forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        logits = model(x_cat_dev, x_cont_dev)

    print(f"Model Output Shape: {logits.shape}")

    # Assert output shape
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected model output ({Config.BATCH_SIZE}, 1), got {logits.shape}"

    # Check for numerical instability (NaNs)
    if torch.isnan(logits).any():
        raise ValueError("Model produced NaN values during forward pass.")

    print("Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training and Inference Execution
    # --------------------------------------------------------------------------
    print("\n>>> [Step 4] Executing Training and Inference Pipeline")

    # We use the library's `train_and_predict` function which encapsulates the entire loop.
    # We use `subset_size` to limit the number of batches processed per epoch,
    # ensuring this demo completes in seconds rather than minutes/hours.

    train_and_predict(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        patience=1,
        subset_size=2000,  # Process only ~4 batches per epoch for speed
        device=Config.DEVICE,
        output_submission=Config.OUTPUT_SUBMISSION,
    )

    # --------------------------------------------------------------------------
    # 5. Submission Validation
    # --------------------------------------------------------------------------
    print("\n>>> [Step 5] Validating Submission File")

    if not os.path.exists(Config.OUTPUT_SUBMISSION):
        raise FileNotFoundError(
            f"Submission file not found at {Config.OUTPUT_SUBMISSION}"
        )

    df_sub = pd.read_csv(Config.OUTPUT_SUBMISSION)
    print(f"Loaded submission. Shape: {df_sub.shape}")
    print(df_sub.head(3))

    # Validate Row Count (Test set has 100,000 samples)
    expected_rows = 100000
    assert (
        len(df_sub) == expected_rows
    ), f"Submission has {len(df_sub)} rows, expected {expected_rows}."

    # Validate Columns
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], f"Invalid columns: {df_sub.columns}. Expected ['id', 'target']."

    # Validate IDs (Must match test set IDs)
    # We can check uniqueness and range roughly
    assert df_sub["id"].nunique() == expected_rows, "IDs are not unique."

    # Validate Probabilities
    probs = df_sub["target"]
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError(
            f"Predictions out of range [0, 1]. Range: [{probs.min()}, {probs.max()}]"
        )

    print("Submission file passed all validation checks.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
