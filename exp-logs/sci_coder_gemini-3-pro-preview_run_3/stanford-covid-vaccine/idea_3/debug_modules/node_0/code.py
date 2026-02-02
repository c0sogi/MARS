import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device, MCRMSELoss, format_submission
from library.data import get_dataloaders
from library.model import ConvTransformer
from library.train import train_one_epoch, validate, generate_predictions


def main():
    # 1. Setup and Configuration Overrides
    print("Setting up demonstration configuration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config for speed and demo purposes
    Config.SEED = 42
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for demo

    # Define a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Override cache paths to avoid overwriting main experiment caches
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data_demo.npy")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data_demo.npy")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data_demo.npy")

    # Create directories
    Config.create_dirs()

    # Set reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\n--- Step 1: Data Loading & Verification ---")
    # We set load_cached_data=False to force processing from metadata for this demo
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Verify Train Loader
    try:
        inputs, targets = next(iter(train_loader))
        print(f"Train Batch - Inputs Shape: {inputs.shape}")
        print(f"Train Batch - Targets Shape: {targets.shape}")

        # Assertions
        # Batch size should be 4 (or less if last batch, but here we have 20 samples / 4 = 5 batches)
        assert (
            inputs.shape[0] == Config.BATCH_SIZE
        ), f"Expected batch size {Config.BATCH_SIZE}, got {inputs.shape[0]}"
        # Sequence length 107
        assert inputs.shape[1] == 107, f"Expected seq len 107, got {inputs.shape[1]}"
        # Input channels 14
        assert (
            inputs.shape[2] == 14
        ), f"Expected input channels 14, got {inputs.shape[2]}"
        # Targets 5
        assert targets.shape[2] == 5, f"Expected 5 targets, got {targets.shape[2]}"
        print("Data shapes verified successfully.")
    except StopIteration:
        raise ValueError("Train loader is empty!")

    # 3. Model Initialization & Forward Pass
    print("\n--- Step 2: Model Initialization & Forward Pass ---")
    model = ConvTransformer().to(device)

    # Move dummy batch to device
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs)
    print(f"Model Output Shape: {outputs.shape}")

    # Verify output shape
    assert outputs.shape == (Config.BATCH_SIZE, 107, 5), "Model output shape mismatch."

    # 4. Loss Function Verification
    print("\n--- Step 3: Loss Calculation ---")
    criterion = MCRMSELoss()

    # Slice to scored length (68) as per competition rules
    outputs_scored = outputs[:, : Config.PRED_LEN, :]
    targets_scored = targets[:, : Config.PRED_LEN, :]

    loss = criterion(outputs_scored, targets_scored)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 5. Training Loop Demonstration
    print("\n--- Step 4: Training Loop Demonstration ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch 1 Val Loss:   {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("Model saved.")

    # 6. Inference & Submission
    print("\n--- Step 5: Inference & Submission ---")

    # Generate predictions
    # Note: test_loader in debug mode also has subset size (20)
    preds, ids = generate_predictions(model, test_loader, device)
    print(f"Predictions Shape: {preds.shape}")
    print(f"Number of IDs: {len(ids)}")

    assert preds.shape[0] == len(ids), "Mismatch between predictions and IDs count"
    assert preds.shape[1] == 107, "Prediction sequence length must be 107"
    assert preds.shape[2] == 5, "Prediction targets must be 5"

    # Format submission
    sub_df = format_submission(preds, ids)
    print("Submission DataFrame Head:")
    print(sub_df.head(2))

    # Verify Submission Structure
    expected_rows = len(ids) * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
