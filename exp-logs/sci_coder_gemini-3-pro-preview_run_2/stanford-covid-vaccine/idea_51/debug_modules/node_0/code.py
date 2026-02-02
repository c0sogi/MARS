import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.data import RNAProcessor, RNADataset, get_dataloaders
from library.model import DSPFN
from library.loss import MCRMSELoss
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===\n")

    # 1. Setup and Configuration Overrides for Speed
    print("--- 1. Configuring Environment for Fast Execution ---")
    set_seed(42)

    # Define a temporary directory for this demo
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config to use the demo directory and smaller model/training params
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_cache.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_cache.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_cache.npz")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Reduce Model Complexity for Demo Speed
    Config.BACKBONE_LAYERS = 2
    Config.BACKBONE_DILATIONS = [1, 2]
    Config.BACKBONE_GROWTH_RATE = 16
    Config.LATENT_DIM = 16
    Config.RNN_HIDDEN_SIZE = 16
    Config.FEEDBACK_EMBED_DIM = 8

    # Training Params
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    print("Config patched for speed.")

    # 2. Create Mini Datasets
    print("\n--- 2. Creating Mini Datasets ---")
    # Load original metadata
    train_full = pd.read_csv(Config.TRAIN_CSV)
    val_full = pd.read_csv(Config.VAL_CSV)
    test_full = pd.read_csv(Config.TEST_CSV)

    # Sample 16 rows for train, 8 for val, 8 for test
    mini_train = train_full.head(16).copy()
    mini_val = val_full.head(8).copy()
    mini_test = test_full.head(8).copy()

    # Save to demo directory
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Patch Config paths to point to mini datasets
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    print(f"Mini datasets saved to {DEMO_DIR}")

    # 3. Data Processing Verification
    print("\n--- 3. Verifying Data Processing ---")
    processor = RNAProcessor()

    # Process the mini training dataframe
    # Note: We must reload from CSV or ensure list columns are handled if passing DF directly.
    # The processor.process_data handles parsing stringified lists if they are strings.
    # Since we saved and are about to load via get_dataloaders, we test that flow.

    # We will verify the processor logic directly on the dataframe first
    data_dict = processor.process_data(mini_train, is_test=False)

    # Assertions
    N_samples = len(mini_train)
    seq_len = Config.SEQ_LEN

    assert data_dict["X_seq"].shape == (
        N_samples,
        seq_len,
        Config.VOCAB_SIZE_SEQ,
    ), "X_seq shape mismatch"
    assert data_dict["X_struct"].shape == (
        N_samples,
        seq_len,
        Config.VOCAB_SIZE_STRUCT,
    ), "X_struct shape mismatch"
    assert data_dict["Y"].shape == (N_samples, seq_len, 5), "Target Y shape mismatch"
    assert data_dict["masks"].shape == (N_samples, seq_len), "Masks shape mismatch"

    # Verify Partner Map Logic
    # Take the first sample
    struct = mini_train.iloc[0]["structure"]
    pmap = data_dict["partner_maps"][0]

    # Check a known pair if exists
    if "(" in struct:
        open_idx = struct.find("(")
        # Find corresponding closing bracket manually
        balance = 0
        close_idx = -1
        for k in range(open_idx, len(struct)):
            if struct[k] == "(":
                balance += 1
            elif struct[k] == ")":
                balance -= 1
            if balance == 0:
                close_idx = k
                break

        if close_idx != -1:
            assert (
                pmap[open_idx] == close_idx
            ), f"Partner map logic failed: {open_idx} should pair with {close_idx}"
            assert (
                pmap[close_idx] == open_idx
            ), f"Partner map logic failed: {close_idx} should pair with {open_idx}"

    print("Data Processor verification passed.")

    # 4. DataLoader Verification
    print("\n--- 4. Verifying DataLoaders ---")
    # Force reload to ensure cache is created from mini datasets
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    targets = batch["targets"]
    partner_map = batch["partner_map"]

    # Inputs: (Batch, Channels, Length)
    # Channels = 18 (4 Seq + 3 Struct + 7 Loop + 4 Partner)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.INPUT_CHANNELS,
        Config.SEQ_LEN,
    ), f"Batch input shape incorrect: {inputs.shape}"

    # Targets: (Batch, Length, 5) -> Note: RNADataset returns Y as (L, 5) per item, so batch is (B, L, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Batch target shape incorrect: {targets.shape}"

    print("DataLoader verification passed.")

    # 5. Model Verification
    print("\n--- 5. Verifying Model Architecture ---")
    model = DSPFN().to(Config.DEVICE)

    inputs = inputs.to(Config.DEVICE)
    partner_map = partner_map.to(Config.DEVICE)

    # Forward Pass
    outputs = model(inputs, partner_map)

    # Output: (Batch, 5, Length)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        5,
        Config.SEQ_LEN,
    ), f"Model output shape incorrect: {outputs.shape}"

    # Check Encode/Decode separation
    z = model.encode(inputs)
    assert z.shape == (
        Config.BATCH_SIZE,
        Config.LATENT_DIM,
        Config.SEQ_LEN,
    ), "Encoder output shape incorrect"

    y_prev = torch.zeros((Config.BATCH_SIZE, 5, Config.SEQ_LEN), device=Config.DEVICE)
    y_decoded = model.decode(z, y_prev, partner_map)
    assert y_decoded.shape == (
        Config.BATCH_SIZE,
        5,
        Config.SEQ_LEN,
    ), "Decoder output shape incorrect"

    print("Model architecture verification passed.")

    # 6. Loss Function Verification
    print("\n--- 6. Verifying MCRMSE Loss ---")
    criterion = MCRMSELoss().to(Config.DEVICE)
    targets = targets.to(Config.DEVICE)
    mask = batch["mask"].to(Config.DEVICE)

    # Compute Loss
    loss = criterion(outputs, targets, mask)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify Masking Logic:
    # Create two predictions that differ ONLY in the unscored region (index > 68)
    # The loss should be identical because the tail is masked/ignored.
    pred1 = torch.zeros_like(outputs)
    pred2 = torch.zeros_like(outputs)

    # Add error in scored region (index 0)
    pred1[:, :, 0] = 1.0
    pred2[:, :, 0] = 1.0

    # Add different error in unscored region (index 100)
    # Config.SEQ_SCORED is 68
    pred1[:, :, 100] = 5.0
    pred2[:, :, 100] = 10.0

    # Dummy targets (all zeros)
    dummy_targets = torch.zeros_like(targets)
    # Dummy mask (valid up to 68)
    dummy_mask = torch.zeros((Config.BATCH_SIZE, Config.SEQ_LEN), device=Config.DEVICE)
    dummy_mask[:, : Config.SEQ_SCORED] = 1.0

    loss1 = criterion(pred1, dummy_targets, dummy_mask)
    loss2 = criterion(pred2, dummy_targets, dummy_mask)

    assert torch.isclose(
        loss1, loss2
    ), "Loss function failed to ignore unscored positions."

    print("Loss function verification passed.")

    # 7. Training Loop Verification
    print("\n--- 7. Verifying Training Loop ---")
    trainer = Trainer()

    # Run one epoch
    print("Running train_epoch...")
    train_loss = trainer.train_epoch(train_loader, epoch_idx=1)
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Run validation
    print("Running validation...")
    val_score = trainer.validate(val_loader)
    print(f"Val Score: {val_score:.4f}")
    assert val_score >= 0, "Validation score should be non-negative"

    # Test full fit method (shortened by Config.EPOCHS=1)
    print("Running trainer.fit()...")
    trainer.fit(train_loader, val_loader, epochs=1)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model was not saved."
    print("Training loop verification passed.")

    # 8. Inference and Submission
    print("\n--- 8. Verifying Inference & Submission Generation ---")
    # Load best model
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(Config.DEVICE)
            partner_map = batch["partner_map"].to(Config.DEVICE)
            batch_ids = batch["id"]

            # Forward
            preds = model(inputs, partner_map)  # (B, 5, L)

            # Move to CPU
            preds = preds.cpu().numpy()

            for i, sample_id in enumerate(batch_ids):
                # Transpose to (L, 5) for CSV format
                sample_pred = preds[i].transpose(1, 0)
                predictions.append(sample_pred)
                ids.append(sample_id)

    # Flatten for submission format
    # Submission requires: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for sample_id, pred_matrix in zip(ids, predictions):
        # pred_matrix is (107, 5)
        for seqpos in range(pred_matrix.shape[0]):
            row_id = f"{sample_id}_{seqpos}"
            row_values = pred_matrix[seqpos]
            row_dict = {"id_seqpos": row_id}
            for idx, col in enumerate(cols):
                row_dict[col] = row_values[idx]
            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Check format
    print(f"Generated submission shape: {submission_df.shape}")
    print(submission_df.head(2))

    expected_rows = len(mini_test) * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    # Save to verify file writing
    sub_path = os.path.join(DEMO_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    assert os.path.exists(sub_path), "Submission file not created."

    print("Inference verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
