import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import library modules
from library.config import Config
from library.data_utils import (
    load_or_process_data,
    RNADataset,
    seed_everything,
    preprocess_data,
)
from library.model import DeepStabilizedBiGRU
from library.loss_metrics import MCRMSELoss, calculate_metric_mcrmse

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== RNA Degradation Prediction Pipeline Demo ====")

    # 1. Configuration Overrides for Demo Speed
    # We override the Config class attributes directly to affect the library modules
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npy")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure demo directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # 2. Data Loading & Processing
    print("\n[2] Loading and Processing Data...")

    # Load full training data (fast enough for 1728 rows)
    # We force reprocessing or loading to ensure we have the dict
    train_data_full = load_or_process_data(
        Config.TRAIN_PATH, Config.TRAIN_CACHE, has_targets=True, load_cached_data=True
    )

    # Create a tiny subset for the demo (e.g., 32 samples)
    subset_size = 32
    print(f"    Creating subset of {subset_size} samples for training demo...")

    train_data_subset = {
        "inputs": train_data_full["inputs"][:subset_size],
        "bpp_indices": train_data_full["bpp_indices"][:subset_size],
        "bpp_masks": train_data_full["bpp_masks"][:subset_size],
        "ids": train_data_full["ids"][:subset_size],
        "targets": train_data_full["targets"][:subset_size],
        "target_masks": train_data_full["target_masks"][:subset_size],
    }

    # Verify Shapes
    assert train_data_subset["inputs"].shape == (
        subset_size,
        107,
        14,
    ), "Input shape mismatch"
    assert train_data_subset["targets"].shape == (
        subset_size,
        107,
        5,
    ), "Target shape mismatch"
    print("    Data shapes verified.")

    # Create Dataset and DataLoader
    train_dataset = RNADataset(train_data_subset, mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )

    # 3. Model Instantiation
    print("\n[3] Initializing DeepStabilizedBiGRU Model...")
    model = DeepStabilizedBiGRU().to(device)

    # Verify forward pass with a dummy batch
    dummy_batch = next(iter(train_loader))
    with torch.no_grad():
        d_in = dummy_batch["inputs"].to(device)
        d_idx = dummy_batch["bpp_indices"].to(device)
        d_mask = dummy_batch["bpp_masks"].to(device)
        d_out = model(d_in, d_idx, d_mask)

    assert d_out.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {d_out.shape}"
    print("    Model forward pass successful. Output shape verified.")

    # 4. Metric Logic Verification
    print("\n[4] Verifying Metric Logic (MCRMSE)...")
    # Create synthetic perfect predictions
    # Shape: (Batch, Seq, 5)
    syn_targets = torch.rand(4, 107, 5)
    syn_preds = syn_targets.clone()  # Perfect prediction
    syn_masks = torch.ones(4, 107)

    # Calculate metric using library function
    metric_val = calculate_metric_mcrmse(syn_preds, syn_targets, syn_masks)
    print(f"    Perfect prediction MCRMSE: {metric_val}")
    assert np.isclose(metric_val, 0.0), "Metric for perfect prediction should be 0.0"

    # Calculate with known offset
    # Add 1.0 to all values. RMSE should be 1.0.
    syn_preds_off = syn_targets + 1.0
    metric_val_off = calculate_metric_mcrmse(syn_preds_off, syn_targets, syn_masks)
    print(f"    Offset (+1.0) prediction MCRMSE: {metric_val_off}")
    # Note: calculate_metric_mcrmse filters for specific columns, but since we added 1.0 to all,
    # the RMSE for each column is 1.0, and the mean is 1.0.
    assert np.isclose(metric_val_off, 1.0), "Metric for +1.0 offset should be 1.0"

    # 5. Training Loop Demo
    print("\n[5] Running Training Loop (2 Epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MCRMSELoss()

    model.train()
    for epoch in range(Config.EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            targets = batch["targets"].to(device)
            target_masks = batch["target_masks"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, bpp_indices, bpp_masks)
            loss = criterion(outputs, targets, target_masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"    Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.4f}")

    # Save demo model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"    Model saved to {Config.MODEL_PATH}")

    # 6. Inference and Submission
    print("\n[6] Generating Submission on Test Data...")

    # Load test data
    test_data_full = load_or_process_data(
        Config.TEST_PATH, Config.TEST_CACHE, has_targets=False, load_cached_data=True
    )

    # Subset test data
    test_subset_size = 10
    test_data_subset = {
        "inputs": test_data_full["inputs"][:test_subset_size],
        "bpp_indices": test_data_full["bpp_indices"][:test_subset_size],
        "bpp_masks": test_data_full["bpp_masks"][:test_subset_size],
        "ids": test_data_full["ids"][:test_subset_size],
    }

    test_dataset = RNADataset(test_data_subset, mode="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_masks)
            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)

    # Format submission
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)
    sub_df = sub_df[["id_seqpos"] + target_cols]

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission generated at {Config.SUBMISSION_PATH}")
    print(f"    Submission shape: {sub_df.shape}")

    # Verify submission integrity
    expected_rows = test_subset_size * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
