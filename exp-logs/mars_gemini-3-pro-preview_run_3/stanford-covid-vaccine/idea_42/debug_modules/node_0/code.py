import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import set_seed, scored_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import (
    train_one_epoch,
    validate,
    run_training,
    generate_submission,
    criterion_mcrmse,
)


def main():
    print("=== RNA Degradation Prediction Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config parameters to ensure the demo runs quickly
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo execution

    # Ensure working directory exists (handled by Config import, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducible seed
    set_seed(Config.SEED)
    print("    Configuration updated: DEBUG=True, EPOCHS=2, BATCH_SIZE=8")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Initialize DataLoaders
    # load_cached_data=False forces processing from parquet to verify preprocessing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to inspect
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    adj = batch["adj_indices"]
    targets = batch["targets"]
    ids = batch["id"]

    # Verify Shapes
    # Inputs: (Batch, SeqLen=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {inputs.shape}"

    # Adjacency: (Batch, SeqLen=107)
    assert adj.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Adjacency shape mismatch: {adj.shape}"

    # Targets: (Batch, SeqLen=107, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Target shape mismatch: {targets.shape}"

    print("    Batch structure verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    # Move batch to device
    inputs_dev = inputs.to(device)
    adj_dev = adj.to(device)
    targets_dev = targets.to(device)

    # Forward Pass
    outputs = model(inputs_dev, adj_dev)

    # Check Output Shape: (Batch, SeqLen=107, 5)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Output shape mismatch: {outputs.shape}"

    print("    Forward pass successful. Output shape matches expectations.")

    # -------------------------------------------------------------------------
    # 4. Metric & Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss and Metric Calculation...")

    # Calculate Training Loss (MCRMSE on all 5 columns, sliced to seq_scored)
    loss = criterion_mcrmse(outputs, targets_dev)

    # Calculate Validation Metric (MCRMSE on 3 scored columns, sliced to seq_scored)
    metric = scored_mcrmse(outputs, targets_dev)

    print(f"    Training Loss (All 5 targets): {loss.item():.4f}")
    print(f"    Scored Metric (3 targets):     {metric.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isnan(metric), "Metric is NaN"

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Full Training Routine (Shortened)...")

    # We call the library function run_training() which uses the Config we modified
    run_training()

    # Verify model file was created
    if os.path.exists(Config.MODEL_PATH):
        print(f"    Model saved successfully at: {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Call the library function to generate submission
    generate_submission()

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission saved to: {Config.SUBMISSION_PATH}")
        print(f"    Submission shape: {df_sub.shape}")

        # Check columns
        expected_cols = ["id_seqpos"] + Config.TARGET_COLS
        assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

        # Check rows: Debug subset size * SeqLen (107)
        # Note: In debug mode, test set is also sliced to DEBUG_SUBSET_SIZE
        expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

        print("    Submission file verified successfully.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
