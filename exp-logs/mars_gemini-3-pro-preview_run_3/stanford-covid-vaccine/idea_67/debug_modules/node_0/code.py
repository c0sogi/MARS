import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.data import get_dataloaders
from library.model import DADBiGRUModel
from library.train import train_one_epoch, validate, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # 1. Configuration Override for Demo
    # We modify Config attributes to run a fast, small-scale experiment
    print("[1] Configuring experiment settings...")
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Update paths to use the demo working directory
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    Config.TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "cache", "train_data_debug.npz"
    )
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "cache", "val_data_debug.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "cache", "test_data_debug.npz")

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(os.path.dirname(Config.TRAIN_CACHE), exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Data Loading
    print("\n[2] Loading Data (Debug Mode)...")
    # debug=True loads only 100 samples and does not overwrite main cache
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify DataLoaders
    print("    Verifying DataLoader shapes...")
    try:
        batch = next(iter(train_loader))
        inputs = batch["inputs"]
        targets = batch["targets"]
        pair_indices = batch["pair_indices"]
        pair_masks = batch["pair_masks"]

        # Check shapes
        # Inputs: (Batch, Seq_Len=107, Channels=14)
        assert inputs.shape == (
            Config.BATCH_SIZE,
            107,
            14,
        ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, 107, 14)}, got {inputs.shape}"

        # Targets: (Batch, Seq_Len=107, Targets=5)
        assert targets.shape == (
            Config.BATCH_SIZE,
            107,
            5,
        ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {targets.shape}"

        # Structure features
        assert pair_indices.shape == (Config.BATCH_SIZE, 107)
        assert pair_masks.shape == (Config.BATCH_SIZE, 107)

        print("    DataLoader verification passed.")

    except StopIteration:
        raise Exception("DataLoader is empty!")

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = DADBiGRUModel().to(Config.DEVICE)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    inputs = inputs.to(Config.DEVICE)
    pair_indices = pair_indices.to(Config.DEVICE)
    pair_masks = pair_masks.to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs, pair_indices, pair_masks)

    # Output should be (Batch, Seq_Len, 5)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {outputs.shape}"
    print("    Forward pass successful.")

    # 4. Metric Verification
    print("\n[4] Verifying Metric (MCRMSE)...")
    # Create dummy data
    # Case: True = 0, Pred = 1. Error = 1. RMSE = 1.
    # We test on the first 68 positions (Config.PRED_LEN)
    dummy_true = torch.zeros((4, 107, 5))
    dummy_pred = torch.ones((4, 107, 5))

    # scored_only=True uses columns [0, 1, 3]
    score = MCRMSE(dummy_true, dummy_pred, scored_only=True)

    # Expected: sqrt((0-1)^2) = 1. Average over columns is 1.
    assert (
        abs(score - 1.0) < 1e-5
    ), f"Metric calculation failed. Expected 1.0, got {score}"
    print(f"    Metric check passed. Score for all-ones error: {score}")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_score = validate(model, val_loader, Config.DEVICE)

        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Assertions to ensure training is actually happening
        assert np.isfinite(train_loss), "Training loss is NaN or Infinite"
        assert np.isfinite(val_score), "Validation score is NaN or Infinite"

    # Save model (simulating the checkpointing)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("    Model saved.")

    # 6. Inference & Submission
    print("\n[6] Generating Submission...")

    # Load model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    # Run inference
    test_preds, test_ids = inference(model, test_loader, Config.DEVICE)

    # Verify inference shape
    # 100 samples in debug mode, 107 seq len, 5 targets
    expected_samples = 100
    assert test_preds.shape == (
        expected_samples,
        107,
        5,
    ), f"Inference shape mismatch. Expected {(expected_samples, 107, 5)}, got {test_preds.shape}"

    # Format submission
    print("    Formatting submission DataFrame...")
    N, L, C = test_preds.shape
    flat_preds = test_preds.reshape(-1, C)

    id_seqpos_list = []
    for sample_id in test_ids:
        for i in range(L):
            id_seqpos_list.append(f"{sample_id}_{i}")

    submission_df = pd.DataFrame(
        flat_preds,
        columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
    )
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Verify submission integrity
    assert len(submission_df) == expected_samples * 107
    assert submission_df.shape[1] == 6
    assert "id_seqpos" in submission_df.columns

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
