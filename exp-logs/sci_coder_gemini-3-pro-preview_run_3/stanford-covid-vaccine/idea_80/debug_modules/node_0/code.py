import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.loss import MCRMSELoss
from library.metrics import calculate_competition_metric
from library.train import run_training


def main():
    print("=== RNA Degradation Prediction: Library Usage Demo ===\n")

    # 1. Setup Configuration
    # We use debug=True to limit the number of samples for speed
    config = Config(debug=True)

    # Override specific config parameters for the demo if needed
    config.batch_size = 8  # Small batch size for demo
    config.num_epochs = 1  # Single epoch

    # Set seeds for reproducibility
    seed_everything(config.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Configuration: Debug={config.debug}, Batch Size={config.batch_size}")

    # 2. Data Loading Demonstration
    print("\n--- 1. Data Loading Verification ---")
    # Force reload to demonstrate processing logic (load_cached_data=False)
    # Note: In a real run, keep True to save time.
    train_loader, val_loader = get_dataloaders(config, load_cached_data=False)

    # Fetch a single batch
    inputs, bpp_indices, bpp_masks, targets = next(iter(train_loader))

    print(f"Batch shapes:")
    print(f"  Inputs:      {inputs.shape}")
    print(f"  BPP Indices: {bpp_indices.shape}")
    print(f"  BPP Masks:   {bpp_masks.shape}")
    print(f"  Targets:     {targets.shape}")

    # Assertions to verify data integrity
    assert inputs.shape == (
        config.batch_size,
        config.seq_len,
        config.input_dim,
    ), f"Input shape mismatch. Expected {(config.batch_size, config.seq_len, config.input_dim)}, got {inputs.shape}"
    assert targets.shape == (
        config.batch_size,
        config.seq_len,
        config.num_targets,
    ), f"Target shape mismatch. Expected {(config.batch_size, config.seq_len, config.num_targets)}, got {targets.shape}"
    assert bpp_indices.shape == (
        config.batch_size,
        config.seq_len,
    ), "BPP Indices shape mismatch"
    assert bpp_masks.shape == (
        config.batch_size,
        config.seq_len,
    ), "BPP Masks shape mismatch"

    print("Data loading logic verified.")

    # 3. Model Demonstration
    print("\n--- 2. Model Forward Pass Verification ---")
    model = RNAModel(config).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    bpp_indices = bpp_indices.to(device)
    bpp_masks = bpp_masks.to(device)
    targets = targets.to(device)

    # Forward pass
    preds = model(inputs, bpp_indices, bpp_masks)

    print(f"Predictions shape: {preds.shape}")

    # Assertions
    assert (
        preds.shape == targets.shape
    ), f"Prediction shape {preds.shape} does not match targets {targets.shape}"
    assert torch.isfinite(preds).all(), "Model output contains NaNs or Infs"

    print("Model forward pass verified.")

    # 4. Loss and Metric Demonstration
    print("\n--- 3. Loss and Metric Verification ---")
    criterion = MCRMSELoss()

    # Calculate Loss
    loss = criterion(preds, targets)
    print(f"Calculated MCRMSE Loss: {loss.item():.6f}")

    # Calculate Competition Metric (Scored columns only)
    metric = calculate_competition_metric(preds, targets, config)
    print(f"Calculated Competition Metric: {metric.item():.6f}")

    # Assertions
    assert loss.item() >= 0, "Loss should be non-negative"
    assert metric.item() >= 0, "Metric should be non-negative"
    # The competition metric is a subset of the total loss, usually close in magnitude
    assert isinstance(loss.item(), float), "Loss is not a float"

    print("Loss and metric logic verified.")

    # 5. Full Training Pipeline Demonstration
    print("\n--- 4. Full Pipeline Execution (Train/Val/Inference) ---")
    print("Running training loop for 1 epoch with debug data...")

    # run_training handles the entire lifecycle:
    # - Config init
    # - Data loading
    # - Model training
    # - Validation
    # - Checkpointing
    # - Inference on Test set
    # - Submission file generation

    # We pass debug=True to ensure it uses the small subset of data
    run_training(debug=True, num_epochs=1)

    # Verify submission file was created
    submission_path = os.path.join(config.working_dir, "submission.csv")
    if os.path.exists(submission_path):
        print(f"\nSuccess! Submission file generated at: {submission_path}")
        sub_df = pd.read_csv(submission_path)
        print(f"Submission shape: {sub_df.shape}")

        # Basic check on submission format
        expected_cols = ["id_seqpos"] + config.target_cols
        assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
        assert len(sub_df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
