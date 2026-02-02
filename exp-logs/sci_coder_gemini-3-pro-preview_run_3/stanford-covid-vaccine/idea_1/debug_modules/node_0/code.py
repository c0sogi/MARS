import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import get_data, RNADataset
from library.model import GlobalMLP
from library.engine import train_model, generate_submission


def main():
    print("=== RNA Degradation Prediction Pipeline Demonstration ===")

    # 1. Setup and Configuration Override for Speed
    print("\n[1] Setting up configuration...")
    seed_everything(Config.SEED)

    # Override Config for faster execution during this demo
    Config.EPOCHS = 2
    Config.HIDDEN_LAYERS = [64, 32]  # Smaller architecture
    Config.BATCH_SIZE = 32
    Config.PATIENCE = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading and Processing
    print("\n[2] Loading and processing data...")

    # Load Train
    train_feats, train_targets = get_data("train", load_cached_data=False)
    print(f"    Train Features: {train_feats.shape}, Targets: {train_targets.shape}")

    # Load Val
    val_feats, val_targets = get_data("val", load_cached_data=False)
    print(f"    Val Features:   {val_feats.shape}, Targets: {val_targets.shape}")

    # Load Test
    test_feats, _ = get_data("test", load_cached_data=False)
    print(f"    Test Features:  {test_feats.shape}")

    # Assertions to verify data integrity
    assert (
        train_feats.shape[1] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {train_feats.shape[1]}"
    assert (
        train_targets.shape[1] == Config.OUTPUT_DIM
    ), f"Expected output dim {Config.OUTPUT_DIM}, got {train_targets.shape[1]}"
    assert (
        train_feats.shape[0] == train_targets.shape[0]
    ), "Train feature/target mismatch"

    # 3. Dataset and DataLoader Creation
    print("\n[3] Creating DataLoaders...")
    train_dataset = RNADataset(train_feats, train_targets)
    val_dataset = RNADataset(val_feats, val_targets)
    test_dataset = RNADataset(test_feats, targets=None)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # 4. Metric Logic Verification
    print("\n[4] Verifying Metric Logic (MCRMSE)...")
    # Create dummy data: 2 samples, 2 columns.
    # True: [[1, 1], [2, 2]], Pred: [[1, 2], [2, 3]]
    # Errors: [[0, -1], [0, -1]] -> Sq Errors: [[0, 1], [0, 1]]
    # Mean Sq Error per col: Col0=0, Col1=1
    # RMSE per col: Col0=0, Col1=1
    # Mean RMSE: 0.5
    y_true_dummy = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    y_pred_dummy = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    loss_val = mcrmse(y_true_dummy, y_pred_dummy).item()
    print(f"    Calculated MCRMSE: {loss_val}")
    assert (
        abs(loss_val - 0.5) < 1e-5
    ), f"Metric verification failed. Expected 0.5, got {loss_val}"
    print("    Metric verification passed.")

    # 5. Model Initialization and Training
    print("\n[5] Initializing and Training Model...")

    # The train_model function handles initialization internally, but we need to pass loaders
    # It returns the trained model and history
    model, history = train_model(train_loader, val_loader, device)

    # Verify History
    print(f"    Training History keys: {history.keys()}")
    assert len(history["train_loss"]) > 0, "No training loss recorded."
    assert len(history["val_loss"]) > 0, "No validation loss recorded."
    print(f"    Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"    Final Val Loss:   {history['val_loss'][-1]:.4f}")

    # 6. Prediction and Submission Generation
    print("\n[6] Generating Submission...")

    # Load test metadata for IDs
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    generate_submission(model, test_loader, test_df, device, save_path=submission_path)

    # 7. Verify Submission File
    print("\n[7] Verifying Submission File...")
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission Shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    # Expected rows: Number of test samples * Sequence Length
    expected_rows = len(test_df) * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Column mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
