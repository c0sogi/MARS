import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device, mcrmse_loss, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_epoch, validate, inference


def run_demo():
    print("==== RNA Degradation Prediction Demo ====")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching
    # -------------------------------------------------------------------------
    # Modify Config to run a lightweight, fast demonstration
    print("[1/7] Patching Configuration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16

    # Reduce Model Complexity for Speed
    Config.HIDDEN_DIM = 64  # Original: 384
    Config.NUM_LAYERS = 2  # Original: 4
    Config.CONV_FILTERS = 32  # Original: 256

    # Set Demo Paths
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Clean/Create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"      Working Directory: {Config.WORKING_DIR}")
    print(f"      Epochs: {Config.EPOCHS}, Hidden Dim: {Config.HIDDEN_DIM}")

    # -------------------------------------------------------------------------
    # 2. Setup
    # -------------------------------------------------------------------------
    print("[2/7] Setting up Environment...")
    seed_everything(Config.SEED)
    device = get_device()
    print(f"      Device: {device}")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("[3/7] Initializing Data Loaders...")
    # load_cached_data=False ensures we test the preprocessing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print("      Verifying Data Shapes...")
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    targets = batch["targets"]

    # Check dimensions
    # Inputs: (Batch, Seq_Len, Input_Dim=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, 107, 14)}, got {inputs.shape}"

    # Pair Indices: (Batch, Seq_Len)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Pair indices shape mismatch. Expected {(Config.BATCH_SIZE, 107)}, got {pair_indices.shape}"

    # Targets: (Batch, Pred_Len=68, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        68,
        5,
    ), f"Targets shape mismatch. Expected {(Config.BATCH_SIZE, 68, 5)}, got {targets.shape}"

    print("      Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("[4/7] Initializing Model...")
    model = RNAModel().to(device)

    print("      Running Forward Pass...")
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    targets = targets.to(device)

    preds = model(inputs, pair_indices)

    # Output: (Batch, Seq_Len=107, Targets=5)
    assert preds.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Prediction shape mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {preds.shape}"

    print("      Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("[5/7] Verifying Metric Logic...")
    # Metric should only consider Config.SCORED_COLS: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices: 0, 1, 3. Indices 2 and 4 should be ignored.

    dummy_target = torch.zeros(2, 68, 5)
    dummy_pred = torch.zeros(2, 107, 5)

    # Scenario A: Perfect prediction on scored columns, massive error on unscored
    dummy_pred[:, :68, 2] = 1000.0  # Index 2 (deg_pH10) - Unscored
    dummy_pred[:, :68, 4] = 1000.0  # Index 4 (deg_50C) - Unscored

    score_a = metric_mcrmse(dummy_pred, dummy_target)
    assert (
        score_a == 0.0
    ), f"Metric failed. Expected 0.0, got {score_a}. Metric should ignore unscored columns."

    # Scenario B: Error on one scored column
    # Set error of 3.0 on Index 0. RMSE for col 0 = 3.0.
    # RMSE for col 1, 3 = 0.0.
    # Mean RMSE = (3.0 + 0 + 0) / 3 = 1.0
    dummy_pred[:, :68, 2] = 0.0
    dummy_pred[:, :68, 4] = 0.0
    dummy_pred[:, :68, 0] = 3.0

    score_b = metric_mcrmse(dummy_pred, dummy_target)
    assert (
        abs(score_b - 1.0) < 1e-5
    ), f"Metric calculation failed. Expected 1.0, got {score_b}"

    print("      Metric logic verified.")

    # -------------------------------------------------------------------------
    # 6. Training & Validation Loop
    # -------------------------------------------------------------------------
    print("[6/7] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Train
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"      Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive."

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"      Val MCRMSE: {val_score:.4f}")
    assert val_score >= 0, "Validation score should be non-negative."

    # -------------------------------------------------------------------------
    # 7. Inference & Submission
    # -------------------------------------------------------------------------
    print("[7/7] Running Inference...")
    # Using the current model state (trained for 1 epoch)
    inference(model, test_loader, device)

    print("      Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Rows: 240 test samples * 107 positions = 25680 rows
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check Columns
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

    print(f"      Submission file valid: {sub_df.shape}")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
