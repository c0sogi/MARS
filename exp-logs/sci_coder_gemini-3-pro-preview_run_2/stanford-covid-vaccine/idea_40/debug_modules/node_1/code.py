import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import TAFRDNModel
from library.loss import MCRMSELoss
from library.engine import Engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Setup
    # We use a specific working directory for this demo to avoid cache conflicts
    # and enable debug mode for speed (fewer epochs, smaller dataset subset).
    working_dir = "./working/demo_execution"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    print(f"Initializing Config with output_dir={working_dir} and debug=True...")
    config = Config(debug=True, output_dir=working_dir)

    # Override specific parameters for the demo to ensure it runs very quickly
    config.epochs = 2
    config.batch_size = 4
    config.subset_size = 20  # Only use 20 samples for this demo
    config.num_workers = 0  # Avoid multiprocessing overhead in simple script

    # Set reproducible seed
    set_seed(config.seed)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading and Verification
    print("\n=== Data Loading ===")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Verify Train Loader
    print("Verifying Train Loader...")
    try:
        inputs, partner_indices, targets = next(iter(train_loader))
    except StopIteration:
        raise ValueError("Train loader is empty!")

    # Expected Shapes:
    # Inputs: (Batch, SeqLen, Channels=18)
    # PartnerIndices: (Batch, SeqLen)
    # Targets: (Batch, SeqLen, NumTargets=5)
    print(f"Input shape: {inputs.shape}")
    print(f"Partner Indices shape: {partner_indices.shape}")
    print(f"Targets shape: {targets.shape}")

    assert inputs.shape == (
        config.batch_size,
        config.seq_len,
        18,
    ), f"Unexpected input shape: {inputs.shape}"
    assert partner_indices.shape == (
        config.batch_size,
        config.seq_len,
    ), f"Unexpected partner_indices shape: {partner_indices.shape}"
    assert targets.shape == (
        config.batch_size,
        config.seq_len,
        5,
    ), f"Unexpected targets shape: {targets.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization and Forward Pass
    print("\n=== Model Verification ===")
    model = TAFRDNModel(config).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    print("Running forward pass...")
    # The model returns a list of outputs corresponding to recycling steps
    outputs = model(inputs, partner_indices)

    assert isinstance(outputs, list), "Model output should be a list (recycling steps)."
    assert (
        len(outputs) == config.recycling_steps
    ), f"Expected {config.recycling_steps} outputs, got {len(outputs)}"

    final_pred = outputs[-1]
    print(f"Final prediction shape: {final_pred.shape}")

    assert final_pred.shape == (
        config.batch_size,
        config.seq_len,
        5,
    ), f"Prediction shape mismatch. Expected {(config.batch_size, config.seq_len, 5)}, got {final_pred.shape}"

    print("Model forward pass verified.")

    # 4. Loss Calculation
    print("\n=== Loss Function Verification ===")
    # Convert scored column names to indices
    scored_indices = [config.target_cols.index(col) for col in config.scored_cols]
    criterion = MCRMSELoss(scored_indices=scored_indices, seq_scored=config.pred_len)

    loss = criterion(final_pred, targets)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() >= 0, "Loss should be non-negative."

    print("Loss function verified.")

    # 5. Training Loop (Engine)
    print("\n=== Training Loop Execution ===")
    engine = Engine(config)

    # Train for the configured epochs (2 in debug mode)
    engine.train(train_loader, val_loader)

    # Check if model checkpoint was saved
    if os.path.exists(config.best_model_path):
        print(f"Training successful. Best model saved at: {config.best_model_path}")
    else:
        raise FileNotFoundError("Best model file was not created after training.")

    # 6. Inference and Submission
    print("\n=== Inference Execution ===")
    engine.predict(test_loader)

    if os.path.exists(config.submission_path):
        print(f"Inference successful. Submission saved at: {config.submission_path}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    # 7. Validate Submission Format
    print("\n=== Validating Submission Format ===")
    sub_df = pd.read_csv(config.submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    # Check columns
    expected_cols = ["id_seqpos"] + config.target_cols
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch.\nExpected: {expected_cols}\nGot: {list(sub_df.columns)}"

    # Check row count
    # In debug mode with subset_size=20, test_loader has 20 samples.
    # Total rows = 20 samples * 107 seq_len = 2140 rows.
    expected_rows = config.subset_size * config.seq_len
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check id_seqpos format (e.g., "id_00b436dec_0")
    sample_id_seqpos = sub_df.iloc[0]["id_seqpos"]
    assert (
        "_" in sample_id_seqpos
    ), "id_seqpos format seems incorrect (missing underscore)."

    print("Submission format validated successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
