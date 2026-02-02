import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import get_dataset, RNADataset
from library.model import InterleavedBiGRU
from library.loss import MaskedMSELoss, mcrmse
from library.train import train_one_epoch


def run_demo():
    print("--- Starting RNA Degradation Pipeline Demo ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override paths to use a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "model")
    Config.PREDS_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.MODEL_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Model Hyperparameters for a tiny, fast model
    Config.HIDDEN_DIM = 64  # Reduced from 384
    Config.NUM_LAYERS = 2  # Reduced from 6
    Config.MLP_EXPANSION_FACTOR = 2
    Config.DROPOUT = 0.0

    # Override Training Hyperparameters
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use 0 for simple debugging/demo
    Config.DEVICE = "cpu"  # Force CPU for deterministic demo execution

    # Initialize directories and seeds
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Config: Dim={Config.HIDDEN_DIM}, Layers={Config.NUM_LAYERS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # -------------------------------------------------------------------------
    print("\n[2] Loading and verifying dataset...")

    # Load training data (force reload from source to test processing logic)
    # Note: This reads ./metadata/train.parquet
    train_dataset = get_dataset("train", load_cached_data=False)

    assert isinstance(train_dataset, RNADataset), "get_dataset should return RNADataset"
    print(f"Train dataset size: {len(train_dataset)}")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = {"sequence", "loop_type", "pair_dist", "targets", "mask", "id"}
    assert (
        set(batch.keys()) == expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Sequence: (B, 107)
    seq_shape = batch["sequence"].shape
    assert seq_shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect sequence shape: {seq_shape}"

    # Targets: (B, 107, 3)
    tgt_shape = batch["targets"].shape
    assert tgt_shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Incorrect targets shape: {tgt_shape}"

    # Mask: (B, 107)
    mask_shape = batch["mask"].shape
    assert mask_shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect mask shape: {mask_shape}"

    print("Batch shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Initializing model and running forward pass...")

    model = InterleavedBiGRU().to(Config.DEVICE)

    # Verify model parameter count is small (due to our config override)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {param_count}")

    # Run forward pass
    sequence = batch["sequence"].to(Config.DEVICE)
    loop_type = batch["loop_type"].to(Config.DEVICE)
    pair_dist = batch["pair_dist"].to(Config.DEVICE)

    preds = model(sequence, loop_type, pair_dist)

    # Verify output shape: (B, 107, 3)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Model output shape mismatch: {preds.shape}"

    print("Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Calculation and Optimization Step
    # -------------------------------------------------------------------------
    print("\n[4] Testing loss calculation and optimization...")

    criterion = MaskedMSELoss()
    targets = batch["targets"].to(Config.DEVICE)
    mask = batch["mask"].to(Config.DEVICE)

    # Calculate Loss
    loss = criterion(preds, targets, mask)
    print(f"Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Calculate Metric (MCRMSE)
    metric_val = mcrmse(preds, targets, mask)
    print(f"Calculated MCRMSE: {metric_val.item():.6f}")

    # Optimization Step
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Optimization step completed.")

    # Save dummy model for inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Saved dummy model to {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Formatting
    # -------------------------------------------------------------------------
    print("\n[5] Simulating inference and submission formatting...")

    # Load test dataset
    test_dataset = get_dataset("test", load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Get a test batch
    test_batch = next(iter(test_loader))

    # Reload model to verify saving/loading works
    model_inf = InterleavedBiGRU().to(Config.DEVICE)
    model_inf.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model_inf.eval()

    with torch.no_grad():
        t_seq = test_batch["sequence"].to(Config.DEVICE)
        t_loop = test_batch["loop_type"].to(Config.DEVICE)
        t_dist = test_batch["pair_dist"].to(Config.DEVICE)
        t_ids = test_batch["id"]

        # Predict
        t_preds = model_inf(t_seq, t_loop, t_dist)  # (B, 107, 3)
        t_preds_np = t_preds.cpu().numpy()

    print(f"Inference output shape: {t_preds_np.shape}")

    # Simulate submission formatting for this batch
    submission_rows = []
    batch_size_actual = t_preds_np.shape[0]

    for i in range(batch_size_actual):
        sample_id = t_ids[i]
        sample_pred = t_preds_np[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions (reactivity, deg_Mg_pH10, deg_Mg_50C)
            vals = sample_pred[seqpos]

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": vals[0],
                    "deg_Mg_pH10": vals[1],
                    "deg_pH10": 0.0,
                    "deg_Mg_50C": vals[2],
                    "deg_50C": 0.0,
                }
            )

    df_sub = pd.DataFrame(submission_rows)

    # Verify Submission DataFrame
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    expected_rows = batch_size_actual * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Save sample submission
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Sample submission saved to {Config.SUBMISSION_PATH}")
    print("First 3 rows:")
    print(df_sub.head(3))

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    # Ensure warnings are suppressed
    import warnings

    warnings.filterwarnings("ignore")

    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
