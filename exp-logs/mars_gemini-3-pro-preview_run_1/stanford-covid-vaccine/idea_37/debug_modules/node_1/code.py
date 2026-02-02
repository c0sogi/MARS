import os
import shutil
import torch
import numpy as np
import pandas as pd
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def demo_main():
    # 1. Setup Directories
    # Create a separate directory for this demo execution to avoid clutter
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    print(f"Created demo directory: {demo_dir}")

    # 2. Create Mini Datasets for Speed
    # We read the original metadata files, take a small slice, and save them to the demo dir.
    print("Creating mini-datasets...")
    original_train_path = config.TRAIN_METADATA
    original_val_path = config.VAL_METADATA
    original_test_path = config.TEST_METADATA

    # Read first 20 rows for train, 10 for val, 10 for test
    # This ensures the demo finishes in seconds
    df_train = pd.read_parquet(original_train_path).head(20)
    df_val = pd.read_parquet(original_val_path).head(10)
    df_test = pd.read_parquet(original_test_path).head(10)

    mini_train_path = os.path.join(demo_dir, "mini_train.parquet")
    mini_val_path = os.path.join(demo_dir, "mini_val.parquet")
    mini_test_path = os.path.join(demo_dir, "mini_test.parquet")

    df_train.to_parquet(mini_train_path, index=False)
    df_val.to_parquet(mini_val_path, index=False)
    df_test.to_parquet(mini_test_path, index=False)

    # 3. Patch Configuration
    # Because the modules import constants directly (e.g., `from library.config import EPOCHS`),
    # we must patch the variables in the importing modules' namespaces as well as the config module.
    print("Patching configuration for demo...")

    # Define new paths and params
    new_model_path = os.path.join(demo_dir, "best_model.pth")
    new_sub_path = os.path.join(demo_dir, "submission.csv")
    new_batch_size = 4
    new_epochs = 2

    # Patch library.config
    config.TRAIN_METADATA = mini_train_path
    config.VAL_METADATA = mini_val_path
    config.TEST_METADATA = mini_test_path
    config.WORKING_DIR = demo_dir

    # Patch library.data (uses metadata paths and working dir)
    data.TRAIN_METADATA = mini_train_path
    data.VAL_METADATA = mini_val_path
    data.TEST_METADATA = mini_test_path
    data.WORKING_DIR = demo_dir

    # Patch library.train (uses hyperparameters and paths)
    train.WORKING_DIR = demo_dir
    train.MODEL_PATH = new_model_path
    train.SUBMISSION_PATH = new_sub_path
    train.BATCH_SIZE = new_batch_size
    train.EPOCHS = new_epochs

    # 4. Demonstrate Data Loading
    print("\n--- Testing Data Loading ---")
    # Load Train Data (load_cached_data=False forces reprocessing from our mini parquets)
    train_ds = data.get_dataset("train", load_cached_data=False)
    print(f"Train Dataset loaded. Size: {len(train_ds)}")

    # Verification
    assert len(train_ds) == 20, "Train dataset size mismatch"

    # Check item structure
    sample = train_ds[0]
    required_keys = ["seq", "loop", "pair_dist", "target", "error"]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset item"

    # Check tensor shapes
    # seq: [107], target: [68, 3]
    assert sample["seq"].shape == (
        config.SEQ_LEN,
    ), f"Seq shape mismatch: {sample['seq'].shape}"
    assert sample["target"].shape == (
        config.SEQ_SCORED,
        3,
    ), f"Target shape mismatch: {sample['target'].shape}"

    # Load Test Data (should have no targets)
    test_ds = data.get_dataset("test", load_cached_data=False)
    print(f"Test Dataset loaded. Size: {len(test_ds)}")
    assert len(test_ds) == 10, "Test dataset size mismatch"
    assert "target" not in test_ds[0], "Test dataset should not have targets"

    # 5. Demonstrate Model Architecture
    print("\n--- Testing Model Architecture ---")
    utils.seed_everything(42)
    device = torch.device("cpu")  # Use CPU for simple demo

    net = model.RNANet().to(device)

    # Create a dummy batch
    dummy_batch_size = 2
    dummy_seq = torch.zeros((dummy_batch_size, config.SEQ_LEN), dtype=torch.long).to(
        device
    )
    dummy_loop = torch.zeros((dummy_batch_size, config.SEQ_LEN), dtype=torch.long).to(
        device
    )
    dummy_dist = torch.zeros((dummy_batch_size, config.SEQ_LEN), dtype=torch.float).to(
        device
    )

    # Forward pass
    pred_val, pred_unc = net(dummy_seq, dummy_loop, dummy_dist)

    print(f"Model Output Shapes: Values {pred_val.shape}, Uncertainty {pred_unc.shape}")

    # Assertions
    expected_shape = (dummy_batch_size, config.SEQ_LEN, 3)
    assert (
        pred_val.shape == expected_shape
    ), f"Expected value shape {expected_shape}, got {pred_val.shape}"
    assert (
        pred_unc.shape == expected_shape
    ), f"Expected unc shape {expected_shape}, got {pred_unc.shape}"

    # 6. Demonstrate Loss Function
    print("\n--- Testing Loss Function ---")
    criterion = model.HomoscedasticLoss().to(device)

    # Dummy targets (only for scored positions)
    dummy_target = torch.randn((dummy_batch_size, config.SEQ_SCORED, 3)).to(device)
    dummy_error = torch.randn((dummy_batch_size, config.SEQ_SCORED, 3)).to(device)

    # Slice predictions to scored length (68)
    pred_val_scored = pred_val[:, : config.SEQ_SCORED, :]
    pred_unc_scored = pred_unc[:, : config.SEQ_SCORED, :]

    loss, mse_val, mse_unc = criterion(
        pred_val_scored, dummy_target, pred_unc_scored, dummy_error
    )

    print(f"Loss: {loss.item():.4f}, MSE Val: {mse_val.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    # 7. Demonstrate Full Training Pipeline
    print("\n--- Running Full Training Pipeline (Mini) ---")
    # This uses the patched config in library.train to run training, validation, and prediction
    # It will use the mini datasets we created and run for 2 epochs.
    train.run_training()

    # 8. Verify Submission Output
    print("\n--- Verifying Submission ---")
    if not os.path.exists(new_sub_path):
        raise FileNotFoundError(f"Submission file not found at {new_sub_path}")

    sub_df = pd.read_csv(new_sub_path)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: 10 test samples * 107 positions = 1070 rows
    expected_rows = 10 * config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check for required columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    for col in expected_cols:
        assert col in sub_df.columns, f"Missing column {col}"

    # Check that non-predicted columns are 0 (deg_pH10, deg_50C) as per the implementation in library/train.py
    assert (sub_df["deg_pH10"] == 0).all(), "deg_pH10 should be 0"
    assert (sub_df["deg_50C"] == 0).all(), "deg_50C should be 0"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo_main()
