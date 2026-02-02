import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders, get_test_loader
from library.model import InterleavedBiGRU
from library.engine import train_model, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("==== RNA Degradation Prediction Pipeline Demo ====")

    # 1. Configure for Fast Demonstration
    print("\n[1] Configuring environment...")

    # Modify Config for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Run for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Main process only for simplicity

    # Ensure working directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Data Loading
    print("\n[2] Loading Data...")

    # Get training and validation loaders
    # Note: The first run might take a moment to process and cache .npz files
    train_loader, val_loader = get_loaders(load_cached_data=True)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")

    # Verify Data Shapes
    batch = next(iter(train_loader))
    x, pair_indices, y, ids = batch["x"], batch["pair_indices"], batch["y"], batch["id"]

    print(
        f"    Batch X shape: {x.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, {Config.INPUT_CHANNELS}])"
    )
    print(
        f"    Batch Pair Indices shape: {pair_indices.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}])"
    )
    print(
        f"    Batch Y shape: {y.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_SCORED}, {Config.NUM_TARGETS}])"
    )

    # Assertions
    assert x.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    ), "Input feature shape mismatch"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "Pair indices shape mismatch"
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), "Target shape mismatch"
    assert len(ids) == Config.BATCH_SIZE, "ID list length mismatch"

    # 3. Model Initialization
    print("\n[3] Initializing Model...")

    model = InterleavedBiGRU()
    model.to(Config.DEVICE)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    with torch.no_grad():
        # Move inputs to device
        x_dev = x.to(Config.DEVICE)
        pair_dev = pair_indices.to(Config.DEVICE)

        preds = model(x_dev, pair_dev)

    print(
        f"    Output shape: {preds.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, {Config.NUM_TARGETS}])"
    )

    # Assert output shape
    # Note: Model outputs predictions for the full sequence length (107),
    # while targets are only for the scored length (68). Slicing happens in loss calculation.
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch"

    # 4. Training
    print("\n[4] Starting Training Loop...")

    # Run the training engine
    # This will save 'best_model.pth' to Config.WORKING_DIR
    train_model(model, train_loader, val_loader)

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("    Training completed successfully.")

    # 5. Inference
    print("\n[5] Running Inference on Test Set...")

    # Get test loader
    test_loader = get_test_loader(load_cached_data=True)

    # Run inference engine
    # This loads 'best_model.pth' and generates 'submission.csv'
    inference(InterleavedBiGRU, test_loader)

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    # 6. Verify Submission Format
    print("\n[6] Verifying Submission Format...")

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Found {list(sub_df.columns)}"

    # Check row count
    # In DEBUG mode, we used a subset of test data (50 samples).
    # Each sample has SEQ_LENGTH (107) rows in the submission.
    # Total rows = 50 * 107 = 5350
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check content of id_seqpos
    sample_id_seqpos = sub_df.iloc[0]["id_seqpos"]
    assert (
        "_0" in sample_id_seqpos or "_1" in sample_id_seqpos
    ), "id_seqpos format seems incorrect"

    print("\n==== Demo Completed Successfully ====")
