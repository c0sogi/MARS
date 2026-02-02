import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import shutil
import warnings

# Import from the provided library
from library.config import Config, setup_reproducibility
from library.utils import get_device, mcrmse_loss
from library.data import get_dataloaders
from library.model import GatedSpatialConvBiGRU
from library.train import train_one_epoch, validate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # 1. Setup & Configuration Overrides
    # We override Config attributes to run a fast, isolated demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up configuration. Working directory: {demo_dir}")

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_cache.npy")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 samples for training
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    setup_reproducibility(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading & Verification
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    targets = batch["targets"]
    ids = batch["id"]

    print(
        f"Batch shapes -> Inputs: {inputs.shape}, Pairs: {pair_indices.shape}, Targets: {targets.shape}"
    )

    # Assertions for Data Integrity
    # Inputs: (Batch, 107, 14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    ), f"Incorrect input shape: {inputs.shape}"

    # Targets: (Batch, 68, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), f"Incorrect target shape: {targets.shape}"

    # Verify One-Hot Encoding Logic (First 4 channels are Sequence A,G,U,C)
    # Sum of first 4 channels should be <= 1 (0 for padding if any, though seqs are fixed length here)
    seq_encoding = inputs[0, :, :4]
    assert torch.all(
        seq_encoding.sum(dim=1) <= 1.0001
    ), "Invalid one-hot encoding for sequence"

    # Verify Pair Indices Logic
    # If index i is paired with j, then index j must be paired with i
    sample_pairs = pair_indices[0].numpy()
    for i, j in enumerate(sample_pairs):
        if j != -1:  # If paired
            assert (
                sample_pairs[j] == i
            ), f"Mismatch in pairing logic at indices {i} and {j}"

    print("Data integrity checks passed.")

    # 3. Model Initialization & Forward Pass
    print("\n--- Initializing Model ---")
    model = GatedSpatialConvBiGRU(Config).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    targets = targets.to(device)

    # Forward Pass
    outputs = model(inputs, pair_indices)
    print(f"Model Output Shape: {outputs.shape}")

    # Assert Output Shape: (Batch, 107, 5)
    # Note: Model outputs predictions for full sequence length (107), targets are only provided for 68.
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)}, got {outputs.shape}"

    # 4. Loss Calculation
    print("\n--- Calculating Loss ---")
    # Slice outputs to match scored sequence length
    outputs_scored = outputs[:, : Config.SEQ_SCORED, :]

    loss = mcrmse_loss(outputs_scored, targets)
    print(f"Initial Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # 5. Training Loop Simulation
    print("\n--- Simulating Training Loop ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_mcrmse = validate(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_mcrmse:.4f}"
        )

        # Basic check to ensure model is learning (or at least running correctly)
        assert not np.isnan(train_loss), "Training loss became NaN"
        assert not np.isnan(val_mcrmse), "Validation metric became NaN"

    # Save the "best" model (just the current one for demo)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Model saved to {Config.BEST_MODEL_PATH}")

    # 6. Inference & Submission
    print("\n--- Generating Submission ---")
    # Reload model to verify loading mechanism
    loaded_model = GatedSpatialConvBiGRU(Config).to(device)
    loaded_model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=device, weights_only=True)
    )

    generate_submission(loaded_model, test_loader, device, Config.SUBMISSION_FILE)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {list(sub_df.columns)}")

    # Expected rows: 240 test samples * 107 positions = 25680
    # Note: If test set size in metadata is different, this number changes.
    # Based on description, test.json has 240 lines.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, found {len(sub_df)}"

    # Check required columns
    required_cols = ["id_seqpos"] + Config.TARGET_COLS
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission"

    # Check values are numeric
    assert pd.api.types.is_numeric_dtype(
        sub_df["reactivity"]
    ), "Reactivity column should be numeric"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
