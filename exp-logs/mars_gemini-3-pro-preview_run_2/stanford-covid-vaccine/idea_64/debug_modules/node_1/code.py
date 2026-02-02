import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Filter warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, MCRMSE_Metric
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import ML_GFN
from library.train import train_one_epoch, validate


def main():
    print("=== Starting RNA Degradation Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1/7] Configuring environment for demo...")

    # Set reproducible seed
    seed_everything(42)

    # Override Config for a lightweight demo
    # We use a specific subdirectory in 'working' to avoid conflicts
    Config.IDEA_NAME = "demo_task"
    Config.IDEA_DIR = os.path.join(Config.WORKING_DIR, Config.IDEA_NAME)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Update cache paths to point to the demo directory
    Config.TRAIN_CACHE = os.path.join(Config.IDEA_DIR, "cache_train.npz")
    Config.VAL_CACHE = os.path.join(Config.IDEA_DIR, "cache_val.npz")
    Config.TEST_CACHE = os.path.join(Config.IDEA_DIR, "cache_test.npz")
    Config.BEST_MODEL_PATH = os.path.join(Config.IDEA_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "submission.csv")

    # Reduce compute load for demo
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use 0 workers for simple debugging/demo
    Config.DEVICE = "cpu"  # Force CPU for simple logic verification

    print(f"    Working Directory: {Config.IDEA_DIR}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2/7] Loading and verifying data...")

    # Load data (this will process metadata/train.csv and cache it)
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Assertions to verify data integrity
    inputs = batch["inputs"]
    targets = batch["targets"]
    mask = batch["mask"]
    pair_indices = batch["pair_indices"]
    ids = batch["id"]

    # Expected shapes:
    # Inputs: (B, 107, 18)
    # Targets: (B, 107, 5)
    # Mask: (B, 107)
    # Pair Indices: (B, 107)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        18,
    ), f"Input shape mismatch: {inputs.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        5,
    ), f"Target shape mismatch: {targets.shape}"
    assert mask.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Mask shape mismatch: {mask.shape}"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Pair indices shape mismatch: {pair_indices.shape}"
    assert len(ids) == Config.BATCH_SIZE, "ID list length mismatch"

    print("    Data shapes verified successfully.")
    print(f"    Input Features: {inputs.shape}")
    print(f"    Targets: {targets.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Inference
    # -------------------------------------------------------------------------
    print("\n[3/7] Initializing model and running inference...")

    model = ML_GFN().to(Config.DEVICE)

    # Run a forward pass (Static Encoding + Dynamic Decoding)
    # Pass 1: No previous predictions
    z = model.encode_static(inputs)
    y1 = model.decode_dynamic(z, pair_indices, prev_preds=None)

    # Check output shape: (B, 107, 5)
    assert y1.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        5,
    ), f"Model output shape mismatch: {y1.shape}"

    # Pass 2: With feedback (simulating the iterative refinement)
    y2 = model.decode_dynamic(z, pair_indices, prev_preds=y1)
    assert y2.shape == y1.shape, "Second pass output shape mismatch"

    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[4/7] Verifying Loss Function logic...")

    criterion = MaskedMCRMSELoss().to(Config.DEVICE)

    # Case A: Perfect Prediction
    # If predictions match targets exactly, loss should be ~0
    loss_zero = criterion(targets, targets, mask)
    # Allow small epsilon for float precision
    assert (
        loss_zero.item() < 1e-6
    ), f"Loss for perfect prediction should be 0, got {loss_zero.item()}"

    # Case B: Error in Unscored Columns
    # Scored columns are 0, 1, 3. Unscored are 2, 4.
    # If we add noise to column 2, loss should NOT change (should still be 0)
    targets_noisy_unscored = targets.clone()
    targets_noisy_unscored[:, :, 2] += 10.0  # Add large error to deg_pH10 (unscored)

    loss_unscored = criterion(targets_noisy_unscored, targets, mask)
    assert (
        loss_unscored.item() < 1e-6
    ), f"Loss changed when modifying unscored column! Got {loss_unscored.item()}"

    # Case C: Error in Scored Columns
    # If we add noise to column 0 (reactivity), loss SHOULD increase
    targets_noisy_scored = targets.clone()
    targets_noisy_scored[:, :, 0] += 1.0

    loss_scored = criterion(targets_noisy_scored, targets, mask)
    assert (
        loss_scored.item() > 0.1
    ), f"Loss did not increase significantly for scored column error! Got {loss_scored.item()}"

    print("    Loss function logic (masking & column selection) verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[5/7] Simulating training loop (1 epoch, limited batches)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # We will wrap the loader to only yield 2 batches to save time
    class LimitedLoader:
        def __init__(self, loader, limit=2):
            self.loader = loader
            self.limit = limit

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.limit:
                    break
                yield batch

        def __len__(self):
            return self.limit

    limited_train_loader = LimitedLoader(train_loader, limit=2)

    # Run training step
    train_loss = train_one_epoch(
        model, limited_train_loader, optimizer, criterion, Config.DEVICE
    )

    print(f"    Training simulation complete. Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 6. Validation & Metric Check
    # -------------------------------------------------------------------------
    print("\n[6/7] Verifying validation and metric calculation...")

    limited_val_loader = LimitedLoader(val_loader, limit=2)

    # Run validation
    val_score, val_loss = validate(model, limited_val_loader, criterion, Config.DEVICE)

    print(f"    Validation Score (MCRMSE): {val_score:.4f}")
    print(f"    Validation Loss: {val_loss:.4f}")

    # Check Metric Class directly
    metric = MCRMSE_Metric()
    # Create dummy preds/targets
    # Preds: all 1s, Targets: all 0s. Scored cols: 3.
    # Diff = 1. Squared = 1. Mean = 1. RMSE = 1.
    dummy_preds = torch.ones((10, 68, 5))
    dummy_targets = torch.zeros((10, 68, 5))
    # Mask all valid
    dummy_mask = torch.ones((10, 68))

    metric.update(dummy_preds, dummy_targets, dummy_mask)
    computed_metric = metric.compute()

    # Expected: sqrt(1) = 1.0
    assert (
        abs(computed_metric - 1.0) < 1e-5
    ), f"Metric computation error. Expected 1.0, got {computed_metric}"
    print("    Metric class calculation verified.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation Simulation
    # -------------------------------------------------------------------------
    print("\n[7/7] Simulating submission generation...")

    # Save the current model as "best model" for the simulation
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # Create a limited test loader
    limited_test_loader = LimitedLoader(test_loader, limit=2)

    # Manual inference loop similar to generate_submission in train.py
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in limited_test_loader:
            inputs = batch["inputs"].to(Config.DEVICE)
            pair_indices = batch["pair_indices"].to(Config.DEVICE)
            batch_ids = batch["id"]

            z = model.encode_static(inputs)
            y1 = model.decode_dynamic(z, pair_indices, prev_preds=None)
            y2 = model.decode_dynamic(z, pair_indices, prev_preds=y1)

            preds_list.append(y2.cpu().numpy())
            ids_list.extend(batch_ids)

    all_preds = np.concatenate(preds_list, axis=0)

    # Verify shape of predictions
    # (N_samples, 107, 5)
    expected_samples = len(ids_list)
    assert all_preds.shape == (
        expected_samples,
        107,
        5,
    ), f"Prediction shape mismatch. Expected ({expected_samples}, 107, 5), got {all_preds.shape}"

    # Generate CSV rows
    submission_data = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])
            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Verify DataFrame columns
    expected_cols = ["id_seqpos"] + target_cols
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    # Verify row count
    assert len(submission_df) == expected_samples * 107, "Submission row count mismatch"

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission generated at {Config.SUBMISSION_PATH}")
    print(f"    Rows: {len(submission_df)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
