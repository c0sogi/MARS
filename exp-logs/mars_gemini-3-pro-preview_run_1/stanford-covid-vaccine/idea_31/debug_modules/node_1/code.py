import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import run_training


def run_demo():
    print("Initializing Demo...")

    # 1. Runtime Configuration Overrides for Speed and Isolation
    # We modify the Config class directly to affect the library modules
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Speed optimizations
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Ensure demo directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Configuration set. Working dir: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Component Demonstration: Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Data Loading ---")
    subset_size = 20
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing for demo
        debug_subset=subset_size,
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    sequences = batch["sequences"]
    loop_types = batch["loop_types"]
    pair_dists = batch["pair_dists"]
    targets = batch["targets"]

    print(f"Batch keys: {list(batch.keys())}")

    # Assertions for shapes
    # Sequences: (Batch, Seq_Len=107)
    assert sequences.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Sequence shape mismatch: {sequences.shape}"
    # Targets: (Batch, Pred_Len=68, Num_Targets=3)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch: {targets.shape}"

    print("Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Component Demonstration: Model & Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Model Architecture ---")
    model = RNAModel(config=Config).to(device)

    # Move batch to device
    sequences = sequences.to(device)
    loop_types = loop_types.to(device)
    pair_dists = pair_dists.to(device)
    targets = targets.to(device)

    # Forward pass
    preds = model(sequences, loop_types, pair_dists)

    # Assertions for output
    # Model outputs predictions for the full sequence length (107)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        preds.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {preds.shape}"

    print(f"Model forward pass successful. Output shape: {preds.shape}")

    # -------------------------------------------------------------------------
    # 4. Component Demonstration: Loss Function
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Loss Calculation ---")
    criterion = MaskedMSELoss()

    loss = criterion(preds, targets)

    # Assert loss is a scalar and valid
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"Loss calculation successful. Loss value: {loss.item():.5f}")

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n--- Executing Full Training Pipeline (Mini-Run) ---")
    # run_training handles the loop, validation, and submission generation
    # We pass the debug_subset again to ensure consistency
    run_training(debug_subset=subset_size)

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Submission Output ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: subset_size (test samples) * 107 (seq_len)
    expected_rows = subset_size * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print("Submission file verified successfully.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
