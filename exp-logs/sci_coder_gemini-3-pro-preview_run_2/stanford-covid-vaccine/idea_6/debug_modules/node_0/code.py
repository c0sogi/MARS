import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.data import process_data, RNADataset
from library.model import StackingAwareHybridNet
from library.utils import mcrmse_loss
from library.train import train_one_epoch, validate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("==== STARTING LIBRARY DEMONSTRATION ====\n")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    print("[1] Configuring environment for fast execution...")

    # Override Config defaults to ensure speed and isolation
    Config.CACHE_DIR = "./working/demo_execution/data_cache/"
    Config.SUBMISSION_PATH = "./working/demo_execution/submission.csv"
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.HIDDEN_DIM = 32  # Reduced from 192
    Config.NUM_LAYERS = 2  # Reduced from 12
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.DEVICE = "cpu"  # Force CPU for simple logic verification

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"    Cache Directory: {Config.CACHE_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Processing Demo
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Data Processing (library.data.process_data)...")

    # We use the provided training metadata
    # Note: process_data handles caching internally
    train_inputs, train_targets, train_ids = process_data(
        Config.TRAIN_CSV, mode="train", load_cached_data=False
    )

    # Assertions to verify data integrity
    print(f"    Inputs Shape: {train_inputs.shape}")
    print(f"    Targets Shape: {train_targets.shape}")

    # Expected: [N, 107, 29] for inputs, [N, 107, 5] for targets
    assert len(train_inputs.shape) == 3
    assert train_inputs.shape[1] == Config.SEQ_LEN  # 107
    assert train_inputs.shape[2] == Config.INPUT_DIM  # 29
    assert train_targets.shape[2] == Config.OUTPUT_DIM  # 5
    assert len(train_ids) == len(train_inputs)

    print("    Data processing verification passed.")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader Demo
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Dataset and DataLoader (library.data.RNADataset)...")

    # Create a small subset for the demo to run instantly
    subset_size = 32
    subset_inputs = train_inputs[:subset_size]
    subset_targets = train_targets[:subset_size]
    subset_ids = train_ids[:subset_size]

    dataset = RNADataset(subset_inputs, subset_targets, subset_ids, mode="train")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Fetch one batch
    batch_inputs, batch_targets, batch_ids = next(iter(loader))

    print(f"    Batch Inputs Shape: {batch_inputs.shape}")
    print(f"    Batch Targets Shape: {batch_targets.shape}")

    # Assertions
    assert batch_inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    assert batch_targets.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.OUTPUT_DIM)
    assert isinstance(batch_inputs, torch.Tensor)

    print("    DataLoader verification passed.")

    # ---------------------------------------------------------
    # 4. Model Architecture Demo
    # ---------------------------------------------------------
    print(
        "\n[4] Demonstrating Model Architecture (library.model.StackingAwareHybridNet)..."
    )

    model = StackingAwareHybridNet(Config).to(Config.DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch_inputs.to(Config.DEVICE))

    print(f"    Model Output Shape: {outputs.shape}")

    # Assertions
    # Output should be [Batch, Seq_Len, Output_Dim]
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.OUTPUT_DIM)
    assert not torch.isnan(outputs).any(), "Model produced NaN values"

    print("    Model architecture verification passed.")

    # ---------------------------------------------------------
    # 5. Loss Function Demo
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Loss Calculation (library.utils.mcrmse_loss)...")

    # Create the scoring mask (first 68 positions)
    mask = torch.zeros((1, Config.SEQ_LEN), device=Config.DEVICE)
    mask[:, : Config.PRED_LEN] = 1.0

    loss = mcrmse_loss(outputs, batch_targets.to(Config.DEVICE), mask)

    print(f"    Calculated MCRMSE Loss: {loss.item():.6f}")

    # Assertions
    assert loss.item() >= 0.0, "Loss cannot be negative"
    assert isinstance(loss, torch.Tensor)

    print("    Loss function verification passed.")

    # ---------------------------------------------------------
    # 6. Training Loop Demo
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Training Loop (library.train.train_one_epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Run one epoch of training
    train_loss = train_one_epoch(model, loader, optimizer, Config.DEVICE, mask)
    print(f"    Train Epoch Loss: {train_loss:.6f}")

    # Run validation
    val_loss = validate(model, loader, Config.DEVICE, mask)
    print(f"    Validation MCRMSE: {val_loss:.6f}")

    # Assertions
    assert train_loss > 0
    assert val_loss > 0

    print("    Training loop verification passed.")

    # ---------------------------------------------------------
    # 7. Submission Generation Demo
    # ---------------------------------------------------------
    print(
        "\n[7] Demonstrating Submission Generation (library.train.generate_submission)..."
    )

    # Save the model state first (required by generate_submission)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)

    # Generate submission
    # Note: This will load test.csv, process it, and run inference
    generate_submission(model_path, Config.DEVICE)

    # Verify output file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission File Created: {Config.SUBMISSION_PATH}")
        print(f"    Submission Shape: {sub_df.shape}")
        print(f"    Submission Columns: {sub_df.columns.tolist()}")

        # Check rows: 240 test samples * 107 positions = 25680 rows
        # Note: The provided test.csv has 240 samples.
        expected_rows = 240 * 107
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(sub_df)}"

        # Check columns
        expected_cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        assert list(sub_df.columns) == expected_cols

        print("    Submission generation verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n==== ALL DEMONSTRATIONS PASSED SUCCESSFULLY ====")


if __name__ == "__main__":
    main()
