import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders, RNADataset
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission


def run_demo():
    print("Initializing Demo Configuration...")

    # 1. Override Config for Speed and Isolation
    Config.EXPERIMENT_ID = "demo_run"
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_ID)
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_SAVE_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Enable Debug mode to use a tiny subset of data (fast execution)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 32  # Small number of samples
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

    # Initialize workspace (creates directories)
    Config.initialize_workspace()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Verify Metric Logic (MCRMSE)
    # =========================================================================
    print("\nVerifying MCRMSE Metric...")
    # Create dummy ground truth and predictions
    # Shape: (Batch=2, Seq=3, Targets=3)
    y_true = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ]
    )
    # Predict exactly +1.0 for every value
    y_pred = y_true + 1.0

    # Squared error is 1.0 everywhere. Mean squared error is 1.0. RMSE is 1.0.
    # Average across columns is 1.0.
    score = MCRMSE(y_true, y_pred)

    assert torch.is_tensor(score), "MCRMSE should return a tensor"
    assert torch.isclose(
        score, torch.tensor(1.0)
    ), f"MCRMSE calculation incorrect. Expected 1.0, got {score.item()}"
    print("MCRMSE check passed.")

    # =========================================================================
    # 3. Verify Data Loading & Processing
    # =========================================================================
    print("\nVerifying Data Loading...")
    # This will process the debug subset and cache it in ./working/demo_run/
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Check keys
    required_keys = ["sequence", "loop_type", "pair_dist", "target", "mask", "id"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check Shapes
    seq = batch["sequence"]
    targets = batch["target"]
    pair_dist = batch["pair_dist"]

    # Sequence should be (Batch, 107)
    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Sequence shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {seq.shape}"

    # Targets should be (Batch, 107, 3)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, 3)}, got {targets.shape}"

    # Pair distance should be (Batch, 107) and float
    assert pair_dist.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair dist shape mismatch"
    assert pair_dist.dtype == torch.float32, "Pair dist should be float32"

    print(f"Data Batch Verified. Sequence shape: {seq.shape}")

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print("\nVerifying Model Architecture...")
    model = RNAModel().to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = batch["loop_type"].to(device)
    pair = pair_dist.to(device)

    # Forward pass
    outputs = model(seq, loop, pair)

    # Check Output Shape: (Batch, 107, 3)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, 3)}, got {outputs.shape}"

    # Check for NaN
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print("Model Forward Pass Verified.")

    # =========================================================================
    # 5. Verify Training Step
    # =========================================================================
    print("\nVerifying Training Step...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    # Capture initial weights of the head to verify update
    initial_head_weight = model.head.weight.data.clone()

    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

    assert isinstance(loss, float), "train_one_epoch should return a float loss"
    assert loss > 0, "Loss should be positive"

    # Check if weights updated
    final_head_weight = model.head.weight.data
    assert not torch.equal(
        initial_head_weight, final_head_weight
    ), "Model weights did not update after training step"

    print(f"Training Step Verified. Loss: {loss:.6f}")

    # =========================================================================
    # 6. Verify Validation
    # =========================================================================
    print("\nVerifying Validation...")
    val_score = validate(model, val_loader, device)

    assert isinstance(val_score, float), "validate should return a float score"
    print(f"Validation Verified. MCRMSE: {val_score:.6f}")

    # =========================================================================
    # 7. Verify Submission Generation
    # =========================================================================
    print("\nVerifying Submission Generation...")
    # Ensure no existing submission
    if os.path.exists(Config.SUBMISSION_SAVE_PATH):
        os.remove(Config.SUBMISSION_SAVE_PATH)

    generate_submission(model, test_loader, device)

    assert os.path.exists(
        Config.SUBMISSION_SAVE_PATH
    ), "Submission file was not created"

    # Check content
    df_sub = pd.read_csv(Config.SUBMISSION_SAVE_PATH)
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]

    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check number of rows
    # In debug mode, we have Config.DEBUG_SAMPLES samples in test.
    # Each sample has 107 positions.
    expected_rows = Config.DEBUG_SAMPLES * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check if ignored columns are 0
    assert (df_sub["deg_pH10"] == 0).all(), "deg_pH10 should be 0"
    assert (df_sub["deg_50C"] == 0).all(), "deg_50C should be 0"

    print("Submission Generation Verified.")

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
