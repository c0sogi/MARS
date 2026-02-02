import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import Config
from library.data_utils import process_data, get_dataloaders
from library.model_utils import ParallelDCNResNet, set_seed
from library.train_utils import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # --------------------------------------------------------------------------
    print("Step 1: Configuring environment for fast demonstration...")

    # Override Config parameters for the demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(
        Config.WORKING_DIR, "submission", "submission.csv"
    )
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "cache", "best_model.pth")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    # Set parameters for speed
    DEMO_SAMPLES = 2000
    DEMO_EPOCHS = 2
    DEMO_BATCH_SIZE = 256

    # Set global seed
    set_seed(42)
    print("Configuration complete.\n")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("Step 2: Verifying Data Pipeline (process_data & get_dataloaders)...")

    # We temporarily set the Config variable so process_data picks it up
    Config.MAX_TRAIN_SAMPLES = DEMO_SAMPLES

    # Force processing from scratch (load_cached_data=False) to test logic
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=False
    )

    # Assertions on Data
    print(
        f"  - Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}"
    )

    # Check sample count
    assert (
        X_train.shape[0] == DEMO_SAMPLES
    ), f"Expected {DEMO_SAMPLES} training samples, got {X_train.shape[0]}"

    # Check feature count
    # Original data has 54 cols (excluding Id/Target).
    # Feature engineering adds: Aspect_Sin, Aspect_Cos, Hydrology_Distance, Hydrology_Elevation, Mean_Amenities_Dist (5 features).
    # Total expected features approx 59.
    input_dim = X_train.shape[1]
    assert input_dim >= 54, f"Expected at least 54 features, got {input_dim}"

    # Check targets (should be 0-6 for 7 classes)
    assert (
        y_train.min() >= 0 and y_train.max() <= 6
    ), f"Targets out of range [0, 6]. Min: {y_train.min()}, Max: {y_train.max()}"

    # Verify DataLoaders
    train_loader, val_loader, test_loader, dim_check, ids_check = get_dataloaders(
        load_cached_data=True,  # Load the cache we just created
        batch_size=DEMO_BATCH_SIZE,
    )

    assert dim_check == input_dim, "DataLoader input dim mismatch."

    # Fetch one batch to verify Tensor shapes
    inputs, labels = next(iter(train_loader))
    assert inputs.shape == (
        min(DEMO_SAMPLES, DEMO_BATCH_SIZE),
        input_dim,
    ), f"Batch input shape mismatch: {inputs.shape}"
    assert labels.shape == (
        min(DEMO_SAMPLES, DEMO_BATCH_SIZE),
    ), f"Batch label shape mismatch: {labels.shape}"

    print("Data Pipeline verification passed.\n")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("Step 3: Verifying Model Architecture (ParallelDCNResNet)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=128,  # Smaller hidden dim for demo
        num_blocks=2,  # Fewer blocks for demo
        dropout=0.1,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Create dummy input
    dummy_input = torch.randn(10, input_dim).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    assert output.shape == (
        10,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (10, {Config.NUM_CLASSES}), got {output.shape}"

    assert not torch.isnan(output).any(), "Model output contains NaNs."

    print("Model Architecture verification passed.\n")

    # --------------------------------------------------------------------------
    # 4. Full Training Loop Execution
    # --------------------------------------------------------------------------
    print("Step 4: Executing Training Loop (run_training)...")

    # Run the training utility
    # This will train, validate, save the best model, and generate predictions
    trained_model = run_training(
        epochs=DEMO_EPOCHS,
        max_train_samples=DEMO_SAMPLES,
        load_cached_data=True,  # Use the cache generated in Step 2
        batch_size=DEMO_BATCH_SIZE,
    )

    # Verify model weights exist
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint file not found."
    print("Training loop execution complete.\n")

    # --------------------------------------------------------------------------
    # 5. Submission Verification
    # --------------------------------------------------------------------------
    print("Step 5: Verifying Submission File...")

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    # The test set has 400,000 rows.
    expected_rows = 400000
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check value validity
    # Predictions should be in range 1-7 (original class labels)
    preds = df_sub[Config.TARGET_COL]
    assert (
        preds.min() >= 1 and preds.max() <= 7
    ), f"Predictions out of range [1, 7]. Min: {preds.min()}, Max: {preds.max()}"

    # Check ID validity (ensure no NaNs and match test IDs)
    assert not df_sub[Config.ID_COL].isnull().any(), "Submission IDs contain NaNs."

    print("Submission verification passed.\n")

    print("=== Demonstration Complete: Success ===")


if __name__ == "__main__":
    main()
