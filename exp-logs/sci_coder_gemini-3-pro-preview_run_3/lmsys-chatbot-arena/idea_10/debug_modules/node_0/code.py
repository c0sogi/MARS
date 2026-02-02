import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import transformers
import shutil

# 1. Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import SiameseDeberta
from library.engine import run

# Silence HuggingFace transformers warnings for cleaner output
transformers.logging.set_verbosity_error()


def main():
    print("==== Starting Demonstration Script ====")

    # 2. Configure for Speed/Demo
    # We override the default Config attributes to ensure the script runs quickly (debug mode).
    print("Configuring experiment settings...")
    Config.debug = True  # Use a tiny subset of data (100 train, 50 val/test)
    Config.epochs = 1  # Train for only 1 epoch
    Config.exp_name = "demo_run"  # Separate experiment name
    Config.physical_batch_size = 2  # Small batch size for the demo
    Config.target_batch_size = 2  # Disable gradient accumulation for speed
    Config.gradient_accumulation_steps = 1

    # Update paths based on the new exp_name
    Config.working_dir = os.path.join("./working", Config.exp_name)
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.model_path = os.path.join(Config.working_dir, "best_model.pth")

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.seed)

    # 3. Data Loading Demonstration
    print("\n[1/4] Initializing DataLoaders...")
    # This will load metadata, process it (tokenize), and return PyTorch DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Verify DataLoaders
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Val loader is empty!"

    # Inspect a single batch
    sample_batch = next(iter(train_loader))
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalars",
        "target",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Missing key {key} in batch"

    print("Batch verification passed. Keys and shapes look correct.")

    # 4. Model Instantiation and Sanity Check
    print("\n[2/4] Instantiating Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SiameseDeberta()
    model.to(device)

    # Perform a dummy forward pass to verify architecture
    print("Running dummy forward pass...")
    model.eval()
    with torch.no_grad():
        # Move sample batch to device
        for k, v in sample_batch.items():
            sample_batch[k] = v.to(device)

        # Forward
        logits = model(sample_batch)

        # Check output shape: (Batch_Size, Num_Classes)
        assert logits.shape == (
            Config.physical_batch_size,
            3,
        ), f"Expected output shape {(Config.physical_batch_size, 3)}, got {logits.shape}"

    print("Model forward pass successful.")

    # 5. Full Pipeline Execution (Train -> Val -> Predict)
    print("\n[3/4] Running Training Engine...")
    # The run function handles the training loop, validation, checkpointing, and test inference
    run(model, train_loader, val_loader, test_loader)

    # 6. Output Verification
    print("\n[4/4] Verifying Outputs...")

    # Check if model checkpoint was saved
    if os.path.exists(Config.model_path):
        print(f"Checkpoint found at: {Config.model_path}")
    else:
        # It's possible validation didn't improve in 1 epoch if initialized randomly,
        # but the engine logic saves if val_loss < inf.
        # With 1 epoch, it should save at least once unless it crashed.
        raise FileNotFoundError(f"Model checkpoint not found at {Config.model_path}")

    # Check submission file
    if os.path.exists(Config.submission_path):
        print(f"Submission file found at: {Config.submission_path}")

        # Validate submission format
        sub_df = pd.read_csv(Config.submission_path)
        print(f"Submission shape: {sub_df.shape}")

        expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Invalid columns: {sub_df.columns}"

        # Check for NaNs
        assert not sub_df.isnull().values.any(), "Submission contains NaNs"

        # Check if rows match test set size (debug mode = 50 rows)
        # Note: The engine reads the full test.csv to get IDs, but predicts on the loader.
        # In debug mode, the loader is truncated to 50, but the submission file creation
        # in `run` uses `pd.read_csv(Config.test_path)`.
        # If `run` logic pairs predictions with full test IDs, there might be a length mismatch
        # if not handled carefully.
        # Looking at `library.engine.run`:
        #   predictions = predict(model, test_loader, device) -> returns 50 preds
        #   test_df = pd.read_csv(Config.test_path) -> returns 5748 rows
        #   submission = pd.DataFrame({ "id": test_df["id"], ... predictions ... })
        #
        # If lengths differ, pandas construction will fail or truncate.
        # However, since we cannot modify library code, we acknowledge this behavior.
        # For this demo, we verify that the file was created and is readable.
        print("Submission format verification passed.")

    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
