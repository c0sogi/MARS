import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library import config, utils, data, model, train


def main():
    print("=== Starting Demo Execution ===")

    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration for demo...")

    # Set seeds
    utils.set_seed(42)

    # Define paths for demo
    DEMO_WORK_DIR = "./working/demo_execution"
    DEMO_META_DIR = "./working/demo_metadata"
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Override config constants to ensure speed and isolation
    config.WORKING_DIR = DEMO_WORK_DIR
    config.CACHE_DIR = DEMO_WORK_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_WORK_DIR, "submission.csv")
    config.EPOCHS = 2
    config.BATCH_SIZE = 4
    config.HIDDEN_DIM = 32  # Reduce model size for speed
    config.GROWTH_RATE = 16

    # 2. Create Data Subsets
    # ---------------------------------------------------------
    print("\n[2] Creating data subsets for speed...")

    def create_subset(src_path, dest_path, n_rows=20):
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            subset = df.head(n_rows).copy()
            subset.to_csv(dest_path, index=False)
            print(f"    Created subset: {dest_path} ({len(subset)} rows)")
            return True
        else:
            print(f"    Warning: Source {src_path} not found.")
            return False

    # Original paths
    orig_train = os.path.join(config.METADATA_DIR, "train.csv")
    orig_val = os.path.join(config.METADATA_DIR, "val.csv")
    orig_test = os.path.join(config.METADATA_DIR, "test.csv")

    # Subset paths
    sub_train = os.path.join(DEMO_META_DIR, "train_subset.csv")
    sub_val = os.path.join(DEMO_META_DIR, "val_subset.csv")
    sub_test = os.path.join(DEMO_META_DIR, "test_subset.csv")

    if create_subset(orig_train, sub_train):
        config.TRAIN_CSV = sub_train
    if create_subset(orig_val, sub_val):
        config.VAL_CSV = sub_val
    if create_subset(orig_test, sub_test):
        config.TEST_CSV = sub_test

    # 3. Verify Data Processing
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Processing logic...")

    # Force processing from scratch by ensuring cache doesn't exist or load_cached_data=False
    # We'll rely on the fact that we changed CACHE_DIR to an empty dir.

    # Process Train Data
    train_data = data.process_data("train", load_cached_data=False)

    # Assertions
    expected_keys = ["X_static", "partner_indices", "y", "ids"]
    for k in expected_keys:
        assert k in train_data, f"Missing key {k} in processed data"

    n_samples = len(train_data["ids"])
    seq_len = config.SEQ_LEN  # 107

    # Check Shapes
    # X_static: [N, 107, 18]
    assert train_data["X_static"].shape == (
        n_samples,
        seq_len,
        18,
    ), f"X_static shape mismatch: {train_data['X_static'].shape}"

    # partner_indices: [N, 107]
    assert train_data["partner_indices"].shape == (
        n_samples,
        seq_len,
    ), f"partner_indices shape mismatch: {train_data['partner_indices'].shape}"

    # y: [N, 107, 5]
    assert train_data["y"].shape == (
        n_samples,
        seq_len,
        5,
    ), f"Targets shape mismatch: {train_data['y'].shape}"

    print("    Data processing shapes verified.")

    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate model
    net = model.RecurrentDenseNet().to(config.DEVICE)

    # Create dummy inputs
    B = 2
    dummy_static = torch.randn(B, seq_len, 18).to(config.DEVICE)
    dummy_recycled = torch.zeros(B, seq_len, 5).to(config.DEVICE)
    # Create valid partner indices (mostly -1, some pairs)
    dummy_partners = torch.full((B, seq_len), -1, dtype=torch.long).to(config.DEVICE)
    # Set a simple pair: 0 <-> 1
    dummy_partners[:, 0] = 1
    dummy_partners[:, 1] = 0

    # Forward pass
    output = net(dummy_static, dummy_recycled, dummy_partners)

    # Check output shape: [B, 107, 5]
    assert output.shape == (
        B,
        seq_len,
        5,
    ), f"Model output shape mismatch. Expected {(B, seq_len, 5)}, got {output.shape}"

    print("    Model forward pass successful.")

    # 5. Verify Loss Function
    # ---------------------------------------------------------
    print("\n[5] Verifying MCRMSE Loss...")

    criterion = utils.MCRMSELoss()

    # Dummy predictions and targets
    # config.SCORED_INDICES are [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    pred = torch.zeros(1, seq_len, 5)
    target = torch.zeros(1, seq_len, 5)

    # Set a known error
    # Index 0 (reactivity): Error = 1.0 -> MSE = 1.0 -> RMSE = 1.0
    pred[0, 0, 0] = 1.0
    target[0, 0, 0] = 0.0

    # Index 1 (deg_Mg_pH10): Error = 2.0 -> MSE = 4.0 -> RMSE = 2.0
    pred[0, 0, 1] = 2.0
    target[0, 0, 1] = 0.0

    # Index 3 (deg_Mg_50C): Error = 0.0 -> RMSE = 0.0

    # Mask: Only first position is valid for simplicity
    mask = torch.zeros(1, seq_len)
    mask[0, 0] = 1.0

    loss = criterion(pred, target, mask)

    # Expected MCRMSE: Mean([1.0, 2.0, 0.0]) = 1.0
    # Note: If mask covers more positions where error is 0, RMSE will decrease.
    # Here mask sum is 1.

    expected_loss = (1.0 + 2.0 + 0.0) / 3.0
    assert (
        abs(loss.item() - expected_loss) < 1e-5
    ), f"Loss calculation mismatch. Expected {expected_loss}, got {loss.item()}"

    print(f"    Loss verified: {loss.item():.4f}")

    # 6. Run Training Loop
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop (Demo)...")

    # Using the train.run_training function
    # It uses config.EPOCHS which we set to 2
    try:
        train.run_training(epochs=config.EPOCHS)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not created."
    print("    Training completed and model saved.")

    # 7. Generate Submission
    # ---------------------------------------------------------
    print("\n[7] Generating Submission...")

    train.generate_submission()

    sub_path = config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(sub_path)
    print(f"    Submission shape: {sub_df.shape}")

    # Expected rows: N_test_samples * 107
    # We used a subset of 20 samples (or however many were in test_subset)
    test_subset_len = len(pd.read_csv(config.TEST_CSV))
    expected_rows = test_subset_len * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    print("    Submission format verified.")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
