import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.engine import run_training


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("Setting up demo configuration...")

    # Create temporary directories for the demo to avoid clutter
    demo_working_dir = "./working/demo_run"
    demo_submission_dir = os.path.join(demo_working_dir, "submission")

    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Monkey-patch Config for a lightweight run
    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = demo_working_dir  # Cache preprocessed data here
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "best_model.pth")

    # Reduce Model Complexity
    Config.HIDDEN_SIZE = 64  # Reduced from 384
    Config.NUM_LAYERS = 2  # Reduced from 6
    Config.EMBED_DIM_SEQ = 16
    Config.EMBED_DIM_LOOP = 16
    Config.EMBED_DIM_DIST = 16

    # Reduce Training Duration
    Config.EPOCHS = 2
    Config.SWA_START_EPOCH = 1  # Start SWA at epoch 1 (0-indexed, so 2nd epoch)
    Config.BATCH_SIZE = 16

    # Set Seed
    seed_everything(Config.SEED)

    print(
        f"Config patched: Epochs={Config.EPOCHS}, Hidden={Config.HIDDEN_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Verify Data Loading
    print("\nVerifying Data Loading...")
    # Force reload to ensure we use the patched cache dir
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Assertions on Data Shapes
    # Batch size might be smaller if dataset < batch_size, but here we expect full batch
    assert batch["seq"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Seq shape mismatch: {batch['seq'].shape}"
    assert batch["loop"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Loop shape mismatch: {batch['loop'].shape}"
    assert batch["dist"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Dist shape mismatch: {batch['dist'].shape}"
    assert batch["targets"].shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {batch['targets'].shape}"

    print("Data Loading verified successfully.")

    # 3. Verify Model Logic
    print("\nVerifying Model Logic...")
    model = RNAModel().to(Config.DEVICE)

    # Move batch to device
    seq = batch["seq"].to(Config.DEVICE)
    loop = batch["loop"].to(Config.DEVICE)
    dist = batch["dist"].to(Config.DEVICE)

    # Forward pass
    outputs = model(seq, loop, dist)

    # Assert Output Shape: (Batch, Seq_Len, Num_Targets)
    # Note: Model outputs for full sequence length (107), targets are only for 68
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    print("Model forward pass verified successfully.")

    # 4. Verify Loss Logic (MaskedMSELoss)
    print("\nVerifying Loss Logic...")
    criterion = MaskedMSELoss()

    # Create dummy predictions and targets
    # Shape: (1, 107, 1) for simplicity in thinking, though code uses (B, 107, 3)
    # We use the actual dimensions: (Batch, Seq_Len, Channels)
    B, L, C = 2, 107, 3
    pred_tensor = torch.zeros((B, L, C))
    target_tensor = torch.zeros(
        (B, L, C)
    )  # Targets usually come as (B, 68, C) but loss handles slicing

    # Case 1: Error in the unscored region (index >= 68)
    # Config.PRED_LEN is 68. Indices 0..67 are scored. Index 68 is NOT scored.
    pred_tensor[0, 68, 0] = 100.0  # Huge error in unscored region
    target_tensor[0, 68, 0] = 0.0

    loss_unscored = criterion(pred_tensor, target_tensor)
    assert (
        loss_unscored.item() == 0.0
    ), f"Loss should be 0.0 for errors outside scored region, got {loss_unscored.item()}"

    # Case 2: Error in scored region (index 0)
    pred_tensor[0, 0, 0] = 1.0
    target_tensor[0, 0, 0] = 0.0
    # MSE for this single element error: (1 - 0)^2 = 1.
    # Averaged over (B * PRED_LEN * C) elements.
    # Total elements = 2 * 68 * 3 = 408.
    # Loss = 1 / 408

    loss_scored = criterion(pred_tensor, target_tensor)
    expected_loss = 1.0 / (B * Config.PRED_LEN * C)

    # Floating point comparison
    assert (
        abs(loss_scored.item() - expected_loss) < 1e-6
    ), f"Loss calculation mismatch. Expected {expected_loss}, got {loss_scored.item()}"

    print("MaskedMSELoss verified successfully.")

    # 5. Run Full Training Pipeline
    print("\nStarting Full Training Pipeline (Demo)...")
    # This calls the provided engine which handles training loop, SWA, and prediction
    run_training(train_loader, val_loader, test_loader)

    # 6. Verify Outputs
    print("\nVerifying Pipeline Outputs...")

    # Check Model File
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    # Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: Number of test samples * Seq Length
    # Test set has 240 samples. Seq len is 107.
    # 240 * 107 = 25680 rows.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
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
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check that unscored columns are 0.0 as per engine logic
    assert (sub_df["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (sub_df["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("Pipeline outputs verified successfully.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
