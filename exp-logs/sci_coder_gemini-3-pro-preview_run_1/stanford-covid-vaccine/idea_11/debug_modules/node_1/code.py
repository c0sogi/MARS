import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library import config
from library import utils
from library import data
from library import model
from library import train


def run_demo():
    # 1. Setup and Configuration Override
    print(">>> Setting up configuration for demo...")

    # Patch the Config class to use a lightweight setting for speed
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    config.Config.WORKING_DIR = demo_working_dir
    config.Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")
    config.Config.CACHE_FILE = os.path.join(demo_working_dir, "processed_data.pt")

    # Reduce model complexity
    config.Config.HIDDEN_DIM = 32
    config.Config.NUM_GRU_LAYERS = 1
    config.Config.NUM_TRANSFORMER_LAYERS = 1
    config.Config.NHEAD = 4
    config.Config.DROPOUT = 0.0

    # Reduce training duration
    config.Config.EPOCHS = 1
    config.Config.BATCH_SIZE = 16
    config.Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed
    utils.seed_everything(config.Config.SEED)
    print("Configuration updated for fast execution.")

    # 2. Verify Utility Functions
    print("\n>>> Verifying Utility Functions (MCRMSE)...")
    # Create synthetic data: Batch=2, Seq=5, Channels=3
    y_true = torch.tensor([[[1.0, 2.0, 3.0]] * 5, [[0.5, 0.5, 0.5]] * 5])
    y_pred = torch.tensor([[[1.1, 1.9, 3.2]] * 5, [[0.6, 0.4, 0.6]] * 5])

    # Calculate using library function
    loss_lib = utils.mcrmse_loss(y_true, y_pred)

    # Calculate manually
    # Squared Error
    se = (y_true - y_pred) ** 2
    # Mean over batch and seq
    mse_per_col = torch.mean(se, dim=(0, 1))
    # RMSE per col
    rmse_per_col = torch.sqrt(mse_per_col)
    # Mean of RMSEs
    loss_manual = torch.mean(rmse_per_col)

    assert torch.isclose(loss_lib, loss_manual), "MCRMSE calculation mismatch!"
    print(f"MCRMSE verification passed. Value: {loss_lib.item():.6f}")

    # 3. Data Processing Demonstration
    print("\n>>> Loading and Processing Data...")
    # Force processing from scratch by ensuring cache doesn't exist (handled by rmtree above)
    # Note: load_and_process_data in library/data.py uses separate cache files per split in WORKING_DIR
    datasets = data.load_and_process_data(load_cached_data=False)

    assert "train" in datasets and "val" in datasets and "test" in datasets
    train_ds = datasets["train"]

    print(f"Train dataset size: {len(train_ds)}")

    # Verify item structure
    item = train_ds[0]
    required_keys = {"seq", "loop", "pair_index", "id", "targets"}
    assert required_keys.issubset(
        item.keys()
    ), f"Missing keys in dataset item: {item.keys()}"

    # Verify shapes
    seq_len = config.Config.SEQ_LENGTH
    assert item["seq"].shape == (
        seq_len,
    ), f"Expected seq shape ({seq_len},), got {item['seq'].shape}"
    assert item["targets"].shape == (
        68,
        3,
    ), f"Expected target shape (68, 3), got {item['targets'].shape}"

    print("Data processing and verification successful.")

    # 4. Model Architecture Verification
    print("\n>>> Verifying Model Architecture...")
    device = torch.device(config.Config.DEVICE)
    net = model.StructureAugmentedHybridNetwork(config=config.Config).to(device)

    # Create a dummy batch
    dataloader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    pair = batch["pair_index"].to(device)

    # Forward pass
    with torch.no_grad():
        output = net(seq, loop, pair)

    # Check output shape: (Batch, Seq_Len, 3)
    expected_shape = (4, seq_len, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print(f"Model forward pass successful. Output shape: {output.shape}")

    # 5. Training Pipeline Execution
    print("\n>>> Running Training Pipeline (1 Epoch)...")
    # This calls library.train.train_model which uses the patched Config
    train.train_model()

    best_model_path = os.path.join(config.Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created!"
    print(f"Training completed. Checkpoint saved at {best_model_path}")

    # 6. Inference and Submission
    print("\n>>> Generating Submission...")
    train.generate_submission()

    submission_path = config.Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created!"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Rows: {len(sub_df)}")

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

    # Verify row count
    # Test set has 240 samples, each length 107. Total rows = 240 * 107 = 25680
    n_test_samples = len(datasets["test"])
    expected_rows = n_test_samples * config.Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Verify unscored columns are 0.0 as per generate_submission logic
    assert (sub_df["deg_pH10"] == 0).all(), "deg_pH10 should be 0.0"
    assert (sub_df["deg_50C"] == 0).all(), "deg_50C should be 0.0"

    print("Submission verification successful.")
    print("\n>>> All demo steps completed successfully.")


if __name__ == "__main__":
    run_demo()
