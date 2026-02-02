import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config, set_seed
from library.dataset import load_data, RNADataset
from library.model import DASR_BiGRU
from library.utils import MCRMSELoss, process_submission
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides for Speed/Demo
    # We modify the Config class directly to isolate this run and make it fast.
    print("\n[1] Configuring environment for demo run...")

    # Create a specific directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Cache paths to demo directory
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_cache.npy")

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Small subset for demonstration
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")

    # Load Train Data
    train_dataset = load_data("train", load_cached_data=False, debug=True)

    # Assertions to verify dataset structure
    assert isinstance(train_dataset, RNADataset), "load_data should return RNADataset"
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} samples, got {len(train_dataset)}"

    # Check item structure
    # Returns: feat, pair_idx, dist, targets, mask
    sample = train_dataset[0]
    feat, pair_idx, dist, targets, mask = sample

    print(
        f"Feature shape: {feat.shape} (Expected: {Config.SEQ_LEN}, {Config.NUM_NODE_FEATURES})"
    )
    print(
        f"Targets shape: {targets.shape} (Expected: {Config.SEQ_LEN}, {Config.NUM_TARGETS})"
    )

    assert feat.shape == (Config.SEQ_LEN, Config.NUM_NODE_FEATURES)
    assert pair_idx.shape == (Config.SEQ_LEN,)
    assert dist.shape == (Config.SEQ_LEN,)
    assert targets.shape == (Config.SEQ_LEN, Config.NUM_TARGETS)
    assert mask.shape == (Config.SEQ_LEN,)

    print("Data loading verification passed.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = DASR_BiGRU().to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    dummy_feat = torch.randn(batch_size, Config.SEQ_LEN, Config.NUM_NODE_FEATURES).to(
        device
    )
    dummy_pair = torch.zeros(batch_size, Config.SEQ_LEN, dtype=torch.long).to(device)
    dummy_dist = torch.zeros(batch_size, Config.SEQ_LEN, dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_feat, dummy_pair, dummy_dist)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        batch_size,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Output shape mismatch. Expected {(batch_size, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {outputs.shape}"

    print("Model architecture verification passed.")

    # 4. Loss Function Verification
    print("\n[4] Verifying MCRMSE Loss...")

    criterion = MCRMSELoss()

    # Case 1: Perfect prediction (Loss should be 0)
    loss_zero = criterion(outputs, outputs)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0)
    ), f"Loss should be 0 for identical inputs, got {loss_zero}"

    # Case 2: Known error
    # Create targets exactly 1.0 away from predictions
    dummy_targets = outputs + 1.0
    # MSE = 1.0, RMSE = 1.0, MCRMSE = 1.0
    loss_one = criterion(outputs, dummy_targets)
    assert torch.isclose(
        loss_one, torch.tensor(1.0)
    ), f"Loss should be 1.0, got {loss_one}"

    print("Loss function verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch, Debug Mode)...")

    # We use the run_training function provided in library/train.py
    # It handles data loading internally, but since we set Config.DEBUG=True and modified paths,
    # it will use the settings we defined in step 1.
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        load_cached_data=True,  # We already cached train in step 2, this tests loading cache
    )

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print("Training loop execution successful.")

    # 6. Inference and Submission Generation
    print("\n[6] Running Inference and Generating Submission...")

    # Load Test Data
    test_dataset = load_data("test", load_cached_data=False, debug=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Test dataset returns: feat, pair_idx, dist
            feat, pair_idx, dist = batch
            feat = feat.to(device)
            pair_idx = pair_idx.to(device)
            dist = dist.to(device)

            preds = model(feat, pair_idx, dist)
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0)

    # Generate Submission
    # We need the IDs from the dataset to format the submission correctly
    test_ids = test_dataset.ids

    # Ensure we have predictions for all loaded test samples
    assert len(all_preds) == len(test_ids)

    process_submission(all_preds, test_ids, save_path=Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission shape: {df_sub.shape}")

    # Expected rows: Num_Samples * Seq_Len
    expected_rows = len(test_ids) * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    print("Submission generation verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
