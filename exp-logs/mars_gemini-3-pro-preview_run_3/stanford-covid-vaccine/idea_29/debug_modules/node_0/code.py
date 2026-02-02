import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_dataloaders
from library.model import RNARegressor
from library.train import run_training


def main():
    print("Initializing Configuration...")
    # Initialize config with debug=True to use a subset of data (100 samples)
    # and num_epochs=2 for a quick training demonstration.
    config = Config(debug=True, num_epochs=2)

    # Ensure reproducibility
    set_seed(config.seed)

    print(f"Working directory: {config.working_dir}")
    print(f"Device: {config.device}")

    # =========================================================================
    # 1. Data Pipeline Verification
    # =========================================================================
    print("\n--- Verifying Data Pipeline ---")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Fetch one batch to inspect
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    bpp_indices = batch["bpp_indices"]
    bpp_mask = batch["bpp_mask"]
    targets = batch["targets"]
    ids = batch["ids"]

    # Check batch size (might be smaller if debug subset < batch_size, but usually fixed by drop_last=True in train)
    current_batch_size = inputs.shape[0]
    print(f"Batch size: {current_batch_size}")

    # Verify Shapes
    # Inputs: (B, 107, 14)
    assert inputs.shape == (
        current_batch_size,
        config.seq_length,
        config.input_dim,
    ), f"Input shape mismatch: {inputs.shape}"

    # Adjacency Indices: (B, 107)
    assert bpp_indices.shape == (
        current_batch_size,
        config.seq_length,
    ), f"BPP Indices shape mismatch: {bpp_indices.shape}"

    # Mask: (B, 107)
    assert bpp_mask.shape == (
        current_batch_size,
        config.seq_length,
    ), f"BPP Mask shape mismatch: {bpp_mask.shape}"

    # Targets: (B, 107, 5)
    assert targets.shape == (
        current_batch_size,
        config.seq_length,
        config.num_targets,
    ), f"Targets shape mismatch: {targets.shape}"

    print("Data shapes verified successfully.")

    # =========================================================================
    # 2. Model Architecture Verification
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")
    model = RNARegressor(config).to(config.device)

    # Move batch to device
    inputs = inputs.to(config.device)
    bpp_indices = bpp_indices.to(config.device)
    bpp_mask = bpp_mask.to(config.device)

    # Forward Pass
    outputs = model(inputs, bpp_indices, bpp_mask)

    # Verify Output Shape: (B, 107, 5)
    assert outputs.shape == (
        current_batch_size,
        config.seq_length,
        config.num_targets,
    ), f"Model output shape mismatch: {outputs.shape}"

    # Verify no NaNs
    assert not torch.isnan(outputs).any(), "Model output contains NaNs."

    print("Model forward pass verified successfully.")

    # =========================================================================
    # 3. Metric Logic Verification
    # =========================================================================
    print("\n--- Verifying Metric Logic ---")
    # Create dummy data
    # Shape: (10, 107, 5)
    y_true_dummy = np.random.rand(10, config.seq_length, 5)
    y_pred_dummy = y_true_dummy.copy()  # Perfect predictions

    # Perfect score should be 0.0
    score_perfect = mcrmse_metric(
        y_true_dummy, y_pred_dummy, seq_scored=config.seq_scored
    )
    assert np.isclose(
        score_perfect, 0.0
    ), f"Metric failed on perfect data. Score: {score_perfect}"

    # Offset predictions by 1.0 -> MSE should be 1.0 -> RMSE should be 1.0 -> MCRMSE should be 1.0
    y_pred_offset = y_true_dummy + 1.0
    score_offset = mcrmse_metric(
        y_true_dummy, y_pred_offset, seq_scored=config.seq_scored
    )
    assert np.isclose(
        score_offset, 1.0
    ), f"Metric failed on offset data. Score: {score_offset}"

    print("Metric logic verified successfully.")

    # =========================================================================
    # 4. Full Training Loop Execution
    # =========================================================================
    print("\n--- Executing Training Loop (Debug Mode) ---")
    # This runs training, validation, and generates submission.csv
    run_training(config)

    print("Training execution completed.")

    # =========================================================================
    # 5. Submission Artifact Verification
    # =========================================================================
    print("\n--- Verifying Submission Artifact ---")
    submission_path = config.submission_path

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Found: {df_sub.columns}"

    # In debug mode, we use a subset of test data (100 samples if debug_subset_size=100)
    # However, the loader in `data.py` applies the subset slice.
    # If debug=True, config.debug_subset_size is 100.
    # Total rows = num_samples * seq_length (107).
    # 100 * 107 = 10700 rows.
    expected_rows = config.debug_subset_size * config.seq_length
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for NaN values in predictions
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("Submission artifact verified successfully.")
    print("\nAll demonstrations and verifications passed.")


if __name__ == "__main__":
    main()
