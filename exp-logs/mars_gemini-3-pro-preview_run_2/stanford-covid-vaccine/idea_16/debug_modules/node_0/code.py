import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

# Import from provided library files
from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import MCRMSELoss
from library.model import ScaleAlignedDenseNet
from library.data import get_loaders


def main():
    print("==== Starting Demonstration of RNA Degradation Prediction Pipeline ====\n")

    # 1. Setup & Configuration
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override Config for speed in this demonstration
    print("Adjusting configuration for fast demonstration...")
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    # We will use the existing metadata files which are relatively small (approx 2k samples total),
    # so we don't need to create subsets manually. The data loader caching will handle speed.

    # 2. Data Loading
    print("\n[Step 1] Loading Data...")
    # This will process CSVs into .npz files if not present, or load them if they are.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Inspect a single batch from training data
    try:
        train_batch = next(iter(train_loader))
        inputs, partner_indices, targets = train_batch

        print(f"  Train Batch Loaded.")
        print(f"  Inputs Shape: {inputs.shape} (Batch, Channels, Seq_Len)")
        print(f"  Partner Indices Shape: {partner_indices.shape} (Batch, Seq_Len)")
        print(f"  Targets Shape: {targets.shape} (Batch, Seq_Len, Num_Targets)")

        # Validations
        assert (
            inputs.shape[1] == Config.INPUT_CHANNELS
        ), f"Expected {Config.INPUT_CHANNELS} channels, got {inputs.shape[1]}"
        assert (
            inputs.shape[2] == Config.SEQ_LEN
        ), f"Expected sequence length {Config.SEQ_LEN}, got {inputs.shape[2]}"
        assert (
            targets.shape[2] == Config.NUM_TARGETS
        ), f"Expected {Config.NUM_TARGETS} targets, got {targets.shape[2]}"
        print("  Data shapes verified successfully.")

    except StopIteration:
        raise ValueError("Train loader is empty!")

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    model = ScaleAlignedDenseNet().to(device)
    print(f"  Model {model.__class__.__name__} instantiated.")

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = MCRMSELoss()  # Uses Config.SCORED_TARGET_INDICES by default
    print("  Optimizer and Loss function initialized.")

    # 4. Training Loop Simulation (Single Step)
    print("\n[Step 3] Simulating Training Step...")
    model.train()

    # Move data to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # Forward Pass
    preds = model(inputs, partner_indices)
    print(f"  Prediction Shape: {preds.shape}")

    # Calculate Loss
    loss = criterion(preds, targets)
    print(f"  Calculated Loss: {loss.item():.6f}")

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  Backward pass and optimizer step completed.")

    # 5. Validation & Metric Calculation
    print("\n[Step 4] Validating and Computing Metric...")
    model.eval()
    metric = MCRMSEMetric(scored_indices=Config.SCORED_TARGET_INDICES)

    with torch.no_grad():
        # Fetch a validation batch
        try:
            val_batch = next(iter(val_loader))
            v_inputs, v_pidx, v_targets = val_batch

            v_inputs = v_inputs.to(device)
            v_pidx = v_pidx.to(device)
            v_targets = v_targets.to(
                device
            )  # Keep on device for model, move to cpu for metric if needed

            # Inference
            v_preds = model(v_inputs, v_pidx)

            # Update metric (Metric class handles cpu conversion)
            metric.update(v_preds, v_targets)

            score = metric.compute()
            print(f"  Validation MCRMSE (on one batch): {score:.6f}")

            # Sanity check
            assert score >= 0, "MCRMSE score should be non-negative"

        except StopIteration:
            print("  Validation loader is empty.")

    # 6. Inference on Test Set
    print("\n[Step 5] Running Inference on Test Set...")
    try:
        test_batch = next(iter(test_loader))
        t_inputs, t_pidx, _ = test_batch  # Test targets are placeholders/None

        t_inputs = t_inputs.to(device)
        t_pidx = t_pidx.to(device)

        t_preds = model(t_inputs, t_pidx)

        print(f"  Test Inputs Shape: {t_inputs.shape}")
        print(f"  Test Predictions Shape: {t_preds.shape}")

        # Verify prediction length matches sequence length
        assert t_preds.shape[1] == Config.SEQ_LEN
        assert t_preds.shape[2] == Config.NUM_TARGETS
        print("  Test inference successful.")

    except StopIteration:
        print("  Test loader is empty.")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
