import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import prepare_datasets, VentilatorDataset
from library.model import LANNet
from library.train import MaskedL1Loss, train_model


def main():
    print("=== Starting Ventilator Pressure Prediction Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # --------------------------------------------------------------------------
    print("1. Setting up configuration for fast execution...")

    # Override Config paths to use a demo directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config class attributes dynamically
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_CHECKPOINT = os.path.join(DEMO_DIR, "best_model.pth")
    Config.LAST_MODEL_CHECKPOINT = os.path.join(DEMO_DIR, "last_model.pth")
    Config.SCALER_PATH = os.path.join(DEMO_DIR, "scaler_stats.npz")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Update Cache paths
    Config.TRAIN_CACHE_X = os.path.join(DEMO_DIR, "train_x.npy")
    Config.TRAIN_CACHE_Y = os.path.join(DEMO_DIR, "train_y.npy")
    Config.TRAIN_CACHE_U_OUT = os.path.join(DEMO_DIR, "train_u_out.npy")
    Config.VAL_CACHE_X = os.path.join(DEMO_DIR, "val_x.npy")
    Config.VAL_CACHE_Y = os.path.join(DEMO_DIR, "val_y.npy")
    Config.VAL_CACHE_U_OUT = os.path.join(DEMO_DIR, "val_u_out.npy")
    Config.TEST_CACHE_X = os.path.join(DEMO_DIR, "test_x.npy")
    Config.TEST_CACHE_IDS = os.path.join(DEMO_DIR, "test_ids.npy")
    Config.TEST_CACHE_U_OUT = os.path.join(DEMO_DIR, "test_u_out.npy")

    # Reduce compute load for demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.DEBUG = True  # Important: This triggers the subsampling in data.py

    seed_everything(Config.SEED)
    print("   Configuration updated successfully.\n")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("2. Verifying Data Pipeline...")

    # Load datasets in debug mode (small subset)
    # load_cached_data=False forces processing from scratch to test feature engineering
    train_ds, val_ds, test_ds = prepare_datasets(load_cached_data=False, debug=True)

    # Assertions for Dataset
    assert isinstance(train_ds, VentilatorDataset), "Train dataset is not correct class"
    # Debug mode in data.py takes first 100 breaths
    expected_breaths = 100
    assert (
        len(train_ds) == expected_breaths
    ), f"Expected {expected_breaths} breaths in debug train set, got {len(train_ds)}"

    # Assertions for Data Shapes
    sample_x, sample_y, sample_u_out = train_ds[0]

    # Expected shape: (80, Input_Dim)
    assert sample_x.ndim == 2, "Input features should be 2D (Seq_Len, Features)"
    assert (
        sample_x.shape[0] == 80
    ), f"Sequence length should be 80, got {sample_x.shape[0]}"
    assert (
        sample_x.shape[1] == Config.INPUT_DIM
    ), f"Feature dim should be {Config.INPUT_DIM}, got {sample_x.shape[1]}"

    # Expected target shape: (80,)
    assert sample_y.shape == (
        80,
    ), f"Target shape should be (80,), got {sample_y.shape}"
    assert sample_u_out.shape == (
        80,
    ), f"Control input shape should be (80,), got {sample_u_out.shape}"

    print("   Data Pipeline verified: Shapes and types are correct.\n")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("3. Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for quick unit test
    model = LANNet(config=Config).to(device)
    model.eval()

    # Create a dummy batch: (Batch_Size, Seq_Len, Features)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 80, Config.INPUT_DIM).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Assert output shape: (Batch_Size, Seq_Len)
    assert output.shape == (
        batch_size,
        80,
    ), f"Model output shape mismatch. Expected {(batch_size, 80)}, got {output.shape}"

    print("   Model Architecture verified: Forward pass successful.\n")

    # --------------------------------------------------------------------------
    # 4. Loss and Metric Verification
    # --------------------------------------------------------------------------
    print("4. Verifying Custom Loss and Metric Logic...")

    criterion = MaskedL1Loss()

    # Create synthetic data
    # Case: 2 time steps.
    # Step 0: Inspiratory (u_out=0). Pred=10, Target=12. Error=2.
    # Step 1: Expiratory (u_out=1). Pred=100, Target=0. Error=100 (Should be ignored).

    preds = torch.tensor([10.0, 100.0])
    targets = torch.tensor([12.0, 0.0])
    u_out = torch.tensor([0.0, 1.0])

    # Test Loss Function
    loss = criterion(preds, targets, u_out)
    expected_loss = 2.0  # |10 - 12| / 1 (count)

    assert torch.isclose(
        loss, torch.tensor(expected_loss)
    ), f"Loss calculation failed. Expected {expected_loss}, got {loss.item()}"

    # Test Metric Function
    metric = compute_metric(preds, targets, u_out)
    assert np.isclose(
        metric, expected_loss
    ), f"Metric calculation failed. Expected {expected_loss}, got {metric}"

    print("   Loss and Metric verified: Expiratory phase correctly masked.\n")

    # --------------------------------------------------------------------------
    # 5. Full Training Pipeline Integration Test
    # --------------------------------------------------------------------------
    print("5. Running Full Training Pipeline (Integration Test)...")

    # Run the training loop (which includes validation and submission generation)
    # We use the debug flag and reduced epochs set in Config
    train_model(debug=True, epochs=Config.EPOCHS)

    # Verify outputs
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Best model checkpoint not found."
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    # Verify Submission Format
    submission_df = pd.read_csv(Config.SUBMISSION_FILE)
    assert list(submission_df.columns) == [
        "id",
        "pressure",
    ], "Submission columns are incorrect."
    assert not submission_df.isnull().values.any(), "Submission contains NaN values."

    # Check length: 100 breaths * 80 steps = 8000 rows (in debug mode)
    expected_rows = 100 * 80
    assert (
        len(submission_df) == expected_rows
    ), f"Submission length mismatch. Expected {expected_rows}, got {len(submission_df)}"

    print(
        f"\n   Pipeline execution successful. Submission generated at {Config.SUBMISSION_FILE}"
    )
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
