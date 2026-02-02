import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import library components
# We assume the script is running from the root directory where ./library exists
sys.path.append(".")
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import ParallelDCNResNet, predict_and_submit
from library.train_utils import Trainer


def run_demo():
    print("Initializing Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo
    # --------------------------------------------------------------------------
    # We modify the Config singleton to run a fast, lightweight version of the task.
    # This ensures the script finishes quickly while exercising all code paths.

    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "model.pth")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary subdirectories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute requirements for demo
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 1024  # Larger batch size for speed on small data
    Config.PATIENCE = 1  # Aggressive early stopping

    print(f"Demo Configuration set. Working dir: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading and Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading and Processing Data...")

    # We use a small sample size (e.g., 5000 rows) to verify the pipeline quickly.
    # load_cached_data=False ensures we test the feature engineering logic.
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, debug_sample_size=5000
    )

    # Logic Verification: Data Shapes
    print(f"Data Loaded. Input Dimension: {input_dim}")

    # Verify we have data
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"
    assert len(test_ids) == 5000, f"Expected 5000 test IDs, got {len(test_ids)}"

    # Verify Input Dimension
    # 54 original features + 5 engineered features = 59 features expected
    # (Note: One-hot encoding of Wilderness/Soil is already in the 54 columns of raw data,
    # but let's just verify it's a positive integer consistent with the pipeline)
    assert input_dim > 50, f"Input dimension {input_dim} seems too low."

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[Step 2] Initializing Model...")

    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        dcn_rank=8,  # Reduced rank for demo
        resnet_width=64,  # Reduced width for demo
        dcn_layers=1,  # Reduced depth
        resnet_blocks=1,
    )

    # Logic Verification: Forward Pass
    # Create a dummy batch to check shape compatibility
    dummy_input = torch.randn(10, input_dim)
    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        10,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (10, {Config.NUM_CLASSES}), got {dummy_output.shape}"
    print("Model initialized and forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("\n[Step 3] Starting Training Loop...")

    trainer = Trainer(model)

    # Fit the model
    # This handles training, validation, scheduler stepping, and checkpoint saving
    trained_model = trainer.fit(train_loader, val_loader)

    # Logic Verification: Checkpoint
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
    print("Training complete. Checkpoint verified.")

    # --------------------------------------------------------------------------
    # 5. Prediction and Submission
    # --------------------------------------------------------------------------
    print("\n[Step 4] Generating Predictions...")

    predict_and_submit(trained_model, test_loader, test_ids)

    # Logic Verification: Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify columns
    assert Config.ID_COL in df_sub.columns, f"Missing ID column {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Missing Target column {Config.TARGET_COL}"

    # Verify row count matches test IDs
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {len(df_sub)}"

    # Verify values are within expected range (1-7 for Cover_Type)
    preds = df_sub[Config.TARGET_COL]
    assert (
        preds.min() >= 1 and preds.max() <= 7
    ), "Predictions contain values outside valid class range [1, 7]"

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
