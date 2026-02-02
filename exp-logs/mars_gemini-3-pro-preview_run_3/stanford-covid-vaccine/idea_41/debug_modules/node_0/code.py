import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import load_data, collate_fn
from library.model import DeepDecoupledBiGRU
from library.train import training_loss_fn, train_one_epoch, validate


def run_demo():
    print("Starting RNA Degradation Prediction Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS = 2
    Config.HIDDEN_DIM = 64  # Smaller model
    Config.STEM_FILTERS = 32
    Config.NUM_LAYERS = 2
    Config.WORKING_DIR = "./working/demo_execution"

    # Update paths based on new working dir
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Set seeds
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    # Load Train Data
    # Note: load_cached_data=False forces reprocessing from metadata for this demo
    # to ensure we don't pick up full-size caches from previous runs in other dirs.
    # However, load_data uses Config.WORKING_DIR for caching, which we just changed,
    # so it's safe to leave defaults or explicit.
    train_dataset = load_data("train", load_cached_data=True, debug=Config.DEBUG)

    # Assertions
    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples, got {len(train_dataset)}"

    sample = train_dataset[0]
    print(f"    Sample keys: {sample.keys()}")

    # Check Feature Shapes: (Seq_Len, Channels) -> (107, 14)
    assert sample["features"].shape == (
        Config.SEQ_LEN,
        Config.NUM_INPUT_CHANNELS,
    ), f"Incorrect feature shape: {sample['features'].shape}"

    # Check Target Shapes: (Seq_Len, Targets) -> (107, 5)
    # Note: Targets are padded to 107, though scoring uses fewer.
    assert sample["targets"].shape == (
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Incorrect target shape: {sample['targets'].shape}"

    # DataLoader & Collation
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    batch = next(iter(train_loader))
    print(
        f"    Batch shapes - Features: {batch['features'].shape}, Pair Indices: {batch['pair_indices'].shape}"
    )

    assert batch["features"].shape[0] == Config.BATCH_SIZE
    assert batch["features"].shape[1] == Config.SEQ_LEN
    assert batch["features"].shape[2] == Config.NUM_INPUT_CHANNELS

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[3] Initializing Model & Verifying Forward Pass...")

    model = DeepDecoupledBiGRU().to(device)

    # Move batch to device
    features = batch["features"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    targets = batch["targets"].to(device)

    # Forward Pass
    preds = model(features, pair_indices)
    print(f"    Prediction shape: {preds.shape}")

    # Assert Output Shape: (Batch, Seq_Len, Num_Targets)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        preds.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {preds.shape}"

    # ==========================================
    # 4. Loss Calculation Verification
    # ==========================================
    print("\n[4] Verifying Loss Functions...")

    # Training Loss (Proxy)
    loss_train = training_loss_fn(preds, targets)
    print(f"    Training Loss (MCRMSE Proxy): {loss_train.item():.4f}")
    assert not torch.isnan(loss_train), "Training loss is NaN"

    # Validation Metric (Official MCRMSE)
    # Note: mcrmse_loss expects inputs on the same device
    loss_metric = mcrmse_loss(targets, preds)
    print(f"    Official Metric (MCRMSE): {loss_metric.item():.4f}")
    assert not torch.isnan(loss_metric), "Metric loss is NaN"

    # ==========================================
    # 5. Training Loop Simulation
    # ==========================================
    print("\n[5] Simulating Training Loop...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Load Validation Data
    val_dataset = load_data("val", load_cached_data=True, debug=Config.DEBUG)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn
    )

    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)
        print(
            f"    Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

    # Save Model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"    Model saved to {Config.MODEL_PATH}")
    assert os.path.exists(Config.MODEL_PATH)

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    print("\n[6] Simulating Inference & Submission...")

    # Load Test Data
    test_dataset = load_data("test", load_cached_data=True, debug=Config.DEBUG)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            f_in = batch["features"].to(device)
            p_in = batch["pair_indices"].to(device)
            ids = batch["id"]

            p_out = model(f_in, p_in)
            all_preds.append(p_out.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"    Total Predictions shape: {all_preds.shape}")

    # Format Submission
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_pred = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]
            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)
    print(f"    Submission DataFrame shape: {submission_df.shape}")
    print(f"    First few rows:\n{submission_df.head(3)}")

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    assert os.path.exists(Config.SUBMISSION_FILE)

    # Verify Submission Format
    expected_rows = len(test_dataset) * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(submission_df)}"

    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(submission_df.columns)}"

    print("\n[SUCCESS] Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
