import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device, mcrmse_numpy
from library.data import RNADataset, get_dataloaders
from library.model import PFR_DN
from library.loss import MCRMSELoss
from library.train import train_epoch, validate, generate_submission


def create_subset_data(source_path, dest_path, n_samples=20):
    """Creates a smaller subset of the CSV data for demonstration purposes."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file {source_path} not found.")

    df = pd.read_csv(source_path)
    # Take top n_samples
    subset = df.head(n_samples).copy()
    subset.to_csv(dest_path, index=False)
    print(f"Created subset: {dest_path} with {len(subset)} samples.")


def run_demo():
    print("=== Starting Demo Execution ===\n")

    # 1. Setup & Configuration Override
    # ----------------------------------------------------------------
    # Define a separate directory for this demo to avoid overwriting main experiment files
    DEMO_DIR = os.path.join("./working", "demo_execution")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Working directory: {DEMO_DIR}")

    # Monkey-patch Config to use the demo directory and smaller hyperparameters
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CSV = os.path.join(DEMO_DIR, "train_subset.csv")
    Config.VAL_CSV = os.path.join(DEMO_DIR, "val_subset.csv")
    Config.TEST_CSV = os.path.join(DEMO_DIR, "test_subset.csv")

    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_data.npz")

    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce compute requirements for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Prepare Subset Data
    # ----------------------------------------------------------------
    # We assume the metadata files exist as per the problem description
    original_train = "./metadata/train.csv"
    original_val = "./metadata/val.csv"
    original_test = "./metadata/test.csv"

    create_subset_data(original_train, Config.TRAIN_CSV, n_samples=20)
    create_subset_data(original_val, Config.VAL_CSV, n_samples=8)
    create_subset_data(original_test, Config.TEST_CSV, n_samples=8)

    # 3. Verify Dataset Loading
    # ----------------------------------------------------------------
    print("\n--- Verifying RNADataset ---")
    # This will trigger processing and caching since cache files don't exist yet
    train_ds = RNADataset(mode="train", load_cached_data=True)

    # Check item structure
    feats, p_indices, targets = train_ds[0]

    # Assertions
    assert feats.shape == (
        Config.SEQ_LENGTH,
        Config.INPUT_DIM,
    ), f"Expected feature shape ({Config.SEQ_LENGTH}, {Config.INPUT_DIM}), got {feats.shape}"
    assert p_indices.shape == (
        Config.SEQ_LENGTH,
    ), f"Expected partner indices shape ({Config.SEQ_LENGTH},), got {p_indices.shape}"
    assert targets.shape == (
        Config.SEQ_LENGTH,
        Config.OUTPUT_DIM,
    ), f"Expected target shape ({Config.SEQ_LENGTH}, {Config.OUTPUT_DIM}), got {targets.shape}"

    print("Dataset verification passed: Shapes are correct.")

    # 4. Verify Model Architecture
    # ----------------------------------------------------------------
    print("\n--- Verifying PFR_DN Model ---")
    model = PFR_DN().to(device)

    # Create dummy batch
    batch_size = 2
    dummy_feats = torch.randn(batch_size, Config.SEQ_LENGTH, Config.INPUT_DIM).to(
        device
    )
    dummy_pidx = torch.zeros(batch_size, Config.SEQ_LENGTH).long().to(device)

    # Forward Pass 1 (Cold Start)
    out_1 = model(dummy_feats, dummy_pidx, recycling=None)

    assert out_1.shape == (
        batch_size,
        Config.SEQ_LENGTH,
        Config.OUTPUT_DIM,
    ), f"Model output shape mismatch: {out_1.shape}"

    # Forward Pass 2 (Recycling)
    out_2 = model(dummy_feats, dummy_pidx, recycling=out_1)

    assert out_2.shape == (
        batch_size,
        Config.SEQ_LENGTH,
        Config.OUTPUT_DIM,
    ), f"Model recycling output shape mismatch: {out_2.shape}"

    print("Model verification passed: Forward pass successful.")

    # 5. Verify Loss Function
    # ----------------------------------------------------------------
    print("\n--- Verifying MCRMSELoss ---")
    criterion = MCRMSELoss()

    # Config.SCORED_COLS are indices [0, 1, 3] in the 5-dim output
    # Config.SCORED_SEQ_LENGTH is 68

    # Case: Prediction is exactly Target + 1.0 for all values
    # The error is 1.0, squared error is 1.0, MSE is 1.0, RMSE is 1.0
    # MCRMSE (mean of RMSEs) should be 1.0

    dummy_target = torch.zeros(batch_size, Config.SEQ_LENGTH, Config.OUTPUT_DIM)
    dummy_pred = torch.ones(batch_size, Config.SEQ_LENGTH, Config.OUTPUT_DIM)

    loss_val = criterion(dummy_pred, dummy_target)

    # Check against expected value (allow small float error)
    assert (
        abs(loss_val.item() - 1.0) < 1e-5
    ), f"Loss calculation incorrect. Expected 1.0, got {loss_val.item()}"

    print("Loss function verification passed.")

    # 6. Run Training Loop
    # ----------------------------------------------------------------
    print("\n--- Running Training Loop (Demo) ---")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Metric={val_metric:.4f}"
        )

        # Simple assertion to ensure loss is valid (not NaN)
        assert not np.isnan(train_loss), "Training loss is NaN!"
        assert not np.isnan(val_metric), "Validation metric is NaN!"

    # Save the 'best' model (just the last one here for demo)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("Training loop completed.")

    # 7. Inference & Submission
    # ----------------------------------------------------------------
    print("\n--- Running Inference & Submission ---")

    # Load model state
    model.load_state_dict(torch.load(Config.MODEL_PATH))

    # Generate submission
    generate_submission(model, test_loader, device)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print(f"Columns: {sub_df.columns.tolist()}")

    # Expected rows: n_test_samples * seq_length
    # We used 8 test samples, seq_length 107 -> 8 * 107 = 856 rows
    expected_rows = 8 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    print("Submission verification passed.")
    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
