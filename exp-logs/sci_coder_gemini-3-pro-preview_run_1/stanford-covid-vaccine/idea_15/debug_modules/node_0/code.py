import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Import library components
from library.config import Config
from library.dataset import prepare_data, RNADataset
from library.model import RNAModel
from library.utils import set_seed, mcrmse_metric, build_submission_df


def run_demo():
    print("--- Starting Library Demonstration ---")

    # 1. Setup and Configuration
    # We create a lightweight config for the demo to ensure speed.
    class DemoConfig(Config):
        WORKING_DIR = "./working/demo_run"
        HIDDEN_DIM = 32  # Reduced from 384
        NUM_LAYERS = 2  # Reduced from 6
        EMBED_DIM_CHAR = 8
        EMBED_DIM_LOOP = 8
        EMBED_DIM_DIST = 8
        BATCH_SIZE = 4
        EPOCHS = 1
        NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
        SUBMISSION_PATH = "./working/demo_run/demo_submission.csv"

    config = DemoConfig()
    set_seed(config.SEED)

    # Clean up demo directory if exists
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print("Configuration initialized.")

    # 2. Data Preparation and Verification
    print("\n[Step 1] Verifying Data Processing...")

    # Load data (this will use the cache logic in library.dataset)
    datasets = prepare_data(config, load_cached_data=False)

    # Verify dictionary keys
    assert (
        "train" in datasets and "val" in datasets and "test" in datasets
    ), "prepare_data should return dict with train, val, test keys"

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Verify a single item
    sample = train_dataset[0]
    required_keys = {"seq", "loop", "dist", "id", "targets"}
    assert required_keys.issubset(
        sample.keys()
    ), f"Dataset item missing keys. Found: {sample.keys()}"

    # Check shapes
    # seq: (107,), dist: (107,), targets: (68, 3)
    assert sample["seq"].shape == (
        config.SEQ_LENGTH,
    ), f"Seq shape mismatch: {sample['seq'].shape}"
    assert sample["dist"].shape == (
        config.SEQ_LENGTH,
    ), f"Dist shape mismatch: {sample['dist'].shape}"
    assert sample["targets"].shape == (
        config.SEQ_SCORED,
        3,
    ), f"Targets shape mismatch: {sample['targets'].shape}"

    print("Data processing verification passed.")

    # 3. Model Initialization and Forward Pass
    print("\n[Step 2] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple demo verification
    model = RNAModel(config).to(device)

    # Create a DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    batch = next(iter(train_loader))

    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    dist = batch["dist"].to(device)

    # Forward pass
    output = model(seq, loop, dist)

    # Verify output shape: (Batch, Seq_Len, 3)
    expected_shape = (config.BATCH_SIZE, config.SEQ_LENGTH, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"Model output shape verified: {output.shape}")

    # 4. Metric Logic Verification
    print("\n[Step 3] Verifying Metric Calculation...")

    # Create dummy ground truth and predictions
    # Shape: (N=2, Seq=2, Cols=3)
    # Case: Exact match -> MCRMSE should be 0.0
    y_true = np.zeros((2, 2, 3))
    y_pred = np.zeros((2, 2, 3))
    score = mcrmse_metric(y_true, y_pred)
    assert np.isclose(score, 0.0), f"Metric failed on exact match. Got {score}"

    # Case: Constant error of 1.0
    # RMSE of 1.0 is 1.0. Mean of RMSEs is 1.0.
    y_pred_err = np.ones((2, 2, 3))
    score_err = mcrmse_metric(y_true, y_pred_err)
    assert np.isclose(
        score_err, 1.0
    ), f"Metric failed on constant error. Got {score_err}"

    # Case: Mixed error
    # Col 0: error 0 -> RMSE 0
    # Col 1: error 1 -> RMSE 1
    # Col 2: error 2 -> RMSE 2
    # Mean RMSE = (0+1+2)/3 = 1.0
    y_pred_mixed = np.zeros((2, 2, 3))
    y_pred_mixed[:, :, 1] = 1.0
    y_pred_mixed[:, :, 2] = 2.0
    score_mixed = mcrmse_metric(y_true, y_pred_mixed)
    assert np.isclose(
        score_mixed, 1.0
    ), f"Metric failed on mixed error. Got {score_mixed}"

    print("Metric logic verification passed.")

    # 5. Integration: Minimal Training Loop
    print("\n[Step 4] Verifying Training Integration...")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    initial_loss = None

    # Run for a few batches
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break  # Only run 3 batches

        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        preds = model(seq, loop, dist)

        # Slice to scored positions
        preds_scored = preds[:, : config.SEQ_SCORED, :]

        loss = criterion(preds_scored, targets)
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()

        print(f"  Batch {i}: Loss = {loss.item():.6f}")

    assert initial_loss is not None, "Training loop did not execute any batches."
    print("Training integration verification passed.")

    # 6. Submission Generation
    print("\n[Step 5] Verifying Submission Generation...")

    test_dataset = datasets["test"]
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model.eval()
    all_test_preds = []
    all_test_ids = []

    # Run inference on a small subset of test data
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)
            all_test_preds.append(preds.cpu())
            all_test_ids.extend(ids)

    all_test_preds = torch.cat(all_test_preds, dim=0)

    # Build dataframe
    sub_df = build_submission_df(
        all_test_ids, all_test_preds, seq_len=config.SEQ_LENGTH
    )

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Verify row count: num_samples * seq_len
    num_samples = len(all_test_ids)
    expected_rows = num_samples * config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Verify unscored columns are 0
    assert (sub_df["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (sub_df["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    # Save
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission generated and saved to {config.SUBMISSION_PATH}")

    print("\n--- All Demonstrations Passed Successfully ---")


if __name__ == "__main__":
    run_demo()
