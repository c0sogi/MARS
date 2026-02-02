import os
import torch
import numpy as np
import pandas as pd
import shutil
import sys

# Import library components
from library.config import Config
from library.utils import set_seed, MCRMSELoss, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel, generate_submission
from library.train import run_training


def main():
    print("==== RNA Degradation Prediction: Library Usage Demo ====")

    # 1. Setup Demo Configuration
    # We modify the Config class attributes directly to isolate this run.
    print("\n[1] Configuring Demo Environment...")

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_data.npy")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_data.npy")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_data.npy")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Override Hyperparameters for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = (
        1  # Will be overridden by debug=True in run_training, but good practice
    )
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")

    # Load dataloaders (this will process metadata -> cache since cache doesn't exist yet)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Fetch one batch to inspect
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    pair_masks = batch["pair_masks"]
    targets = batch["targets"]

    print(f"Batch keys: {list(batch.keys())}")
    print(
        f"Inputs shape: {inputs.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.INPUT_CHANNELS}])"
    )
    print(
        f"Targets shape: {targets.shape} (Expected: [{Config.BATCH_SIZE}, {Config.PRED_LEN}, {Config.NUM_TARGETS}])"
    )

    # Assertions
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), "Input shape mismatch"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_TARGETS,
    ), "Target shape mismatch"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair indices shape mismatch"
    assert pair_masks.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair masks shape mismatch"

    print("Data loading verification passed.")

    # 3. Model Initialization & Forward Pass
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    # Move batch to device
    inputs_dev = inputs.to(device)
    p_idx_dev = pair_indices.to(device)
    p_mask_dev = pair_masks.to(device)
    targets_dev = targets.to(device)

    # Forward pass
    preds = model(inputs_dev, p_idx_dev, p_mask_dev)

    print(
        f"Predictions shape: {preds.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.NUM_TARGETS}])"
    )

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Prediction shape mismatch"
    print("Model forward pass verification passed.")

    # 4. Loss Calculation Verification
    print("\n[4] Verifying Loss Function (MCRMSE)...")

    criterion = MCRMSELoss()
    loss = criterion(preds, targets_dev)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    # Check metric utility
    metric_val = metric_mcrmse(preds, targets_dev)
    print(f"Metric MCRMSE: {metric_val}")
    assert metric_val >= 0, "Metric should be non-negative"

    print("Loss function verification passed.")

    # 5. Training Loop Execution
    print("\n[5] Running Training Loop (Debug Mode)...")

    # run_training with debug=True runs for 2 epochs with limited batches
    best_score = run_training(epochs=1, debug=True)

    print(f"Training finished. Best Score: {best_score}")

    # Check if model file was created
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH}"
    print("Training loop verification passed.")

    # 6. Inference & Submission
    print("\n[6] Generating Submission...")

    # This function loads the best model from Config.MODEL_PATH and predicts on test_loader
    generate_submission()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame shape: {df_sub.shape}")
    print(f"Submission Columns: {list(df_sub.columns)}")

    # Calculate expected rows: 240 test samples * 107 sequence length
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("Submission verification passed.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
