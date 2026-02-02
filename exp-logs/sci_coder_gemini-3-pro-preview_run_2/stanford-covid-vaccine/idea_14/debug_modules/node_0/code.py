import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, MetricTracker, format_submission
from library.loss import MCRMSELoss
from library.data import get_loaders, get_test_loader
from library.model import HybridNet


def run_demo():
    # 1. Setup and Configuration
    print("--- 1. Setup and Configuration ---")
    seed_everything(42)

    # Override Config for the demonstration to ensure isolation and speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_VERSION = "demo_v1"
    Config.TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, f"train_data_{Config.CACHE_VERSION}.npz"
    )
    Config.VAL_CACHE = os.path.join(
        Config.WORKING_DIR, f"val_data_{Config.CACHE_VERSION}.npz"
    )
    Config.TEST_CACHE = os.path.join(
        Config.WORKING_DIR, f"test_data_{Config.CACHE_VERSION}.npz"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Reduce hyperparameters for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n--- 2. Data Loading ---")
    # Initialize loaders
    train_loader, val_loader = get_loaders(batch_size=Config.BATCH_SIZE, num_workers=0)

    # Fetch a single batch to verify structure
    x_batch, indices_batch, y_batch = next(iter(train_loader))

    # Move to device
    x_batch = x_batch.to(Config.DEVICE)
    y_batch = y_batch.to(Config.DEVICE)
    indices_batch = {k: v.to(Config.DEVICE) for k, v in indices_batch.items()}

    print(f"Batch X shape: {x_batch.shape}")
    print(f"Batch Y shape: {y_batch.shape}")

    # Assertions for Data
    # Input: (Batch, Channels=14, Seq_Len=107)
    assert x_batch.shape == (
        Config.BATCH_SIZE,
        14,
        107,
    ), f"Incorrect input shape: {x_batch.shape}"
    # Target: (Batch, Seq_Len=107, Channels=5)
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Incorrect target shape: {y_batch.shape}"
    # Indices: (Batch, Seq_Len=107)
    assert indices_batch["p"].shape == (
        Config.BATCH_SIZE,
        107,
    ), "Incorrect partner index shape"

    print("Data loading and shape verification passed.")

    # 3. Model Initialization
    print("\n--- 3. Model Initialization ---")
    model = HybridNet().to(Config.DEVICE)

    # Forward pass check
    model.train()
    preds = model(x_batch, indices_batch)

    print(f"Prediction shape: {preds.shape}")

    # Assertions for Model Output
    # Output: (Batch, Seq_Len=107, Channels=5)
    assert preds.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Incorrect prediction shape: {preds.shape}"
    print("Model forward pass verification passed.")

    # 4. Loss Calculation
    print("\n--- 4. Loss Calculation ---")
    criterion = MCRMSELoss()

    # The loss function expects full length tensors but only scores specific columns
    # We slice predictions to match the scored length (68) for the metric calculation logic
    # embedded in mcrmse_loss if we wanted to be exact, but the provided loss handles
    # the column selection. The competition metric is calculated on the first 68 positions.

    # For the purpose of this demo using the provided library:
    # library.config.mcrmse_loss takes (pred, target) and computes MSE on columns [0, 1, 3].
    # It does not internally slice the sequence length, so we must slice before passing if required.
    # Config.PRED_LEN is 68.

    pred_scored = preds[:, : Config.PRED_LEN, :]
    y_scored = y_batch[:, : Config.PRED_LEN, :]

    loss = criterion(pred_scored, y_scored)

    print(f"Calculated Loss: {loss.item():.5f}")

    # Assertions for Loss
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss contains NaN"
    print("Loss calculation verification passed.")

    # 5. Training Loop Simulation
    print("\n--- 5. Training Loop Simulation (5 Batches) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tracker = MetricTracker()

    model.train()
    for i, (x, indices, y) in enumerate(train_loader):
        if i >= 5:
            break  # Limit to 5 batches

        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
        indices = {k: v.to(Config.DEVICE) for k, v in indices.items()}

        optimizer.zero_grad()

        outputs = model(x, indices)

        # Slice for loss calculation (first 68 bases)
        out_scored = outputs[:, : Config.PRED_LEN, :]
        y_scored = y[:, : Config.PRED_LEN, :]

        loss = criterion(out_scored, y_scored)
        loss.backward()
        optimizer.step()

        # Update metric tracker
        # Tracker expects numpy arrays
        tracker.update(out_scored, y_scored)

        print(f"Batch {i+1} processed. Loss: {loss.item():.4f}")

    train_mcrmse = tracker.result()
    print(f"Training MCRMSE (Subset): {train_mcrmse:.4f}")

    # Save dummy model for inference step
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("Model saved.")

    # 6. Inference and Submission
    print("\n--- 6. Inference and Submission ---")
    test_loader = get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=0)

    model.eval()
    all_preds = []
    all_ids = []

    print("Running inference on test set (subset)...")
    with torch.no_grad():
        for i, (x, indices, ids) in enumerate(test_loader):
            if i >= 5:
                break  # Limit to 5 batches

            x = x.to(Config.DEVICE)
            indices = {k: v.to(Config.DEVICE) for k, v in indices.items()}

            out = model(x, indices)

            all_preds.append(out.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)

        print(f"Collected predictions shape: {all_preds.shape}")
        print(f"Collected IDs count: {len(all_ids)}")

        # Format submission
        print("Generating submission file...")
        format_submission(all_preds, all_ids, save_path=Config.SUBMISSION_PATH)

        # Verify submission file
        assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file loaded. Shape: {df_sub.shape}")

        # Expected rows: Num_Samples * 107
        expected_rows = len(all_ids) * 107
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(df_sub)}"

        # Expected columns
        expected_cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        assert list(df_sub.columns) == expected_cols, "Incorrect submission columns"

        print("Submission verification passed.")
    else:
        print("No predictions generated (Test loader might be empty or skipped).")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
