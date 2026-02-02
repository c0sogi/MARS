import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path so we can import from library
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data_utils import get_dataloaders, get_test_ids
from library.model_utils import ParallelDCNResNet
from library.train_utils import run_training, generate_predictions


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config parameters for a fast demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.LEARNING_RATE = 1e-3

    # Use smaller model dimensions for speed
    Config.HIDDEN_DIM = 64
    Config.LOW_RANK_FACTOR = 4

    # Define working directories for this run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}")

    # ---------------------------------------------------------
    # 2. Data Loading & Processing
    # ---------------------------------------------------------
    print("\n[2] Processing Data...")

    # We use a small subset (1000 samples) to verify the pipeline quickly.
    # load_cached_data=False ensures we test the raw data processing logic.
    MAX_SAMPLES = 1000

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers to avoid overhead in this small demo
        load_cached_data=False,
        max_samples=MAX_SAMPLES,
    )

    # Verify DataLoaders
    try:
        X_batch, y_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    input_dim = X_batch.shape[1]
    print(f"    Batch Shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"    Detected Input Features: {input_dim}")

    # Assertions
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert X_batch.shape[0] <= Config.BATCH_SIZE, "Batch size exceeds limit"
    assert not torch.isnan(X_batch).any(), "Input data contains NaNs"

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Instantiate model manually to check shapes
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        low_rank_factor=Config.LOW_RANK_FACTOR,
        num_cross_layers=2,
        num_res_blocks=1,
    ).to(Config.DEVICE)

    # Run a dummy forward pass
    with torch.no_grad():
        dummy_input = X_batch.to(Config.DEVICE)
        dummy_output = model(dummy_input)

    print(f"    Model Output Shape: {dummy_output.shape}")

    # Assertions
    assert dummy_output.shape == (
        X_batch.shape[0],
        Config.NUM_CLASSES,
    ), f"Expected output shape {(X_batch.shape[0], Config.NUM_CLASSES)}, got {dummy_output.shape}"

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop...")

    # run_training handles model instantiation internally using Config values.
    # Since we updated Config.HIDDEN_DIM/EPOCHS above, it will use those settings.
    trained_model = run_training(
        train_loader, val_loader, input_dim, Config.NUM_CLASSES
    )

    assert trained_model is not None, "Training function returned None"

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[5] Generating Predictions...")

    # Get all Test IDs
    all_test_ids = get_test_ids()

    # CRITICAL: Since we subsampled the data using max_samples=1000 in get_dataloaders,
    # the test_loader only contains 1000 samples. We must slice the IDs to match.
    test_ids_subset = all_test_ids[:MAX_SAMPLES]

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    generate_predictions(trained_model, test_loader, test_ids_subset, submission_path)

    # ---------------------------------------------------------
    # 6. Final Validation
    # ---------------------------------------------------------
    print("\n[6] Validating Submission File...")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Columns: {list(df_sub.columns)}")

    # Assertions
    assert (
        df_sub.shape[0] == MAX_SAMPLES
    ), f"Expected {MAX_SAMPLES} rows, got {df_sub.shape[0]}"
    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Missing required columns"
    assert (
        df_sub["Cover_Type"].isnull().sum() == 0
    ), "Submission contains null predictions"
    assert df_sub["Id"].dtype in [
        np.int64,
        np.int32,
        int,
    ], "Id column should be integer"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
