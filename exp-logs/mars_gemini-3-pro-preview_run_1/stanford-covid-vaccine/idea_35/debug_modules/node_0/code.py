import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.data import get_data, RNADataset, collate_fn
from library.model import RNAModel
from library.utils import set_seed, masked_mse_loss
from library.train import train_model, generate_submission
from torch.utils.data import DataLoader


def main():
    print("--- Starting RNA Degradation Prediction Demo ---")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # --------------------------------------------------------------------------
    print("Configuring environment...")

    # Create a separate directory for this demo run
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Patch the Config class to use the demo directory and lightweight model settings
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir

    # Update derived paths
    Config.CACHE_TRAIN_PATH = os.path.join(demo_dir, "train_data_debug.pt")
    Config.CACHE_VAL_PATH = os.path.join(demo_dir, "val_data_debug.pt")
    Config.CACHE_TEST_PATH = os.path.join(demo_dir, "test_data.pt")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce Model Complexity for Speed
    Config.EMBEDDING_DIM = 16
    Config.HIDDEN_DIM = 32
    Config.N_LAYERS = 1
    Config.DROPOUT = 0.0

    # Training settings
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo

    set_seed(Config.SEED)
    print("Configuration updated for demo run.")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n--- Verifying Data Pipeline ---")

    # Load Training Data
    # This will read from metadata/train.parquet and save to our new cache path
    train_data = get_data(mode="train", load_cached_data=False)

    # Assertions to verify data structure
    assert "seq_tokens" in train_data
    assert "pair_dists" in train_data
    assert "targets" in train_data
    assert "ids" in train_data

    # Check shapes
    n_samples = len(train_data["ids"])
    print(f"Loaded {n_samples} training samples.")

    assert train_data["seq_tokens"].shape == (
        n_samples,
        107,
    ), f"Expected seq_tokens shape ({n_samples}, 107), got {train_data['seq_tokens'].shape}"
    assert train_data["targets"].shape == (
        n_samples,
        107,
        3,
    ), f"Expected targets shape ({n_samples}, 107, 3), got {train_data['targets'].shape}"

    # Instantiate Dataset
    ds = RNADataset(train_data, mode="train")
    assert len(ds) == n_samples

    # Test DataLoader and Collate
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=False)
    batch = next(iter(dl))

    print("Batch keys:", batch.keys())
    assert "seq" in batch
    assert "loop" in batch
    assert "dist" in batch
    assert "target" in batch
    assert batch["seq"].shape == (4, 107)
    assert batch["target"].shape == (4, 107, 3)

    print("Data pipeline verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Verification
    # --------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    model = RNAModel(config=Config).to(Config.DEVICE)

    # Move batch to device
    seq = batch["seq"].to(Config.DEVICE)
    loop = batch["loop"].to(Config.DEVICE)
    dist = batch["dist"].to(Config.DEVICE)
    targets = batch["target"].to(Config.DEVICE)

    # Forward Pass
    preds = model(seq, loop, dist)

    print(f"Model output shape: {preds.shape}")
    assert preds.shape == (
        4,
        107,
        3,
    ), f"Expected output shape (4, 107, 3), got {preds.shape}"

    # Loss Calculation
    loss = masked_mse_loss(preds, targets, scored_len=Config.PRED_LENGTH)
    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n--- Executing Training Loop (1 Epoch) ---")
    # This function uses the Config class we patched earlier
    train_model()

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print("Training loop completed and model saved.")

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    # This function loads the saved model and predicts on test set
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print("First few rows:")
    print(sub_df.head())

    # Basic Checks
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Check row count: 240 test samples * 107 positions = 25680 rows
    # Note: The provided sample_submission.csv has 25680 rows.
    # The test.json has 240 lines.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check that unscored columns are 0.0 as per our logic
    assert (sub_df["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (sub_df["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("Submission verified successfully.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
