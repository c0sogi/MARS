import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import logging

# Filter warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_log_loss, get_logger
from library.data import get_dataloaders
from library.model import SiameseDeberta
from library.engine import run_training, predict_and_submit


def run_demo():
    # 1. Setup and Configuration
    print(">>> Setting up configuration for Demo Run...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for demonstration
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Check device
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading and Verification
    print("\n>>> Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing to demo logic
        debug=True,
        batch_size=Config.TRAIN_BATCH_SIZE,
    )

    # Fetch a single batch to verify structure
    print("Verifying batch structure...")
    batch = next(iter(train_loader))

    # Verify Keys
    expected_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalars",
        "labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Shapes
    # Input IDs: [Batch, SeqLen]
    seq_len = batch["input_ids_a"].shape[1]
    assert batch["input_ids_a"].shape == (Config.TRAIN_BATCH_SIZE, seq_len)
    assert batch["input_ids_b"].shape == (Config.TRAIN_BATCH_SIZE, seq_len)

    # Scalars: [Batch, 3] (log lengths of prompt, resp_a, resp_b)
    assert batch["scalars"].shape == (Config.TRAIN_BATCH_SIZE, 3)

    # Labels: [Batch, 3] (Probabilities for A, B, Tie)
    assert batch["labels"].shape == (Config.TRAIN_BATCH_SIZE, 3)

    print("Batch structure verified successfully.")

    # 3. Model Initialization and Forward Pass
    print("\n>>> Initializing Model...")
    model = SiameseDeberta()
    model.to(Config.DEVICE)

    print("Running dry-run forward pass...")
    # Move batch to device
    device_batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}

    # Forward pass
    outputs = model(
        input_ids_a=device_batch["input_ids_a"],
        attention_mask_a=device_batch["attention_mask_a"],
        input_ids_b=device_batch["input_ids_b"],
        attention_mask_b=device_batch["attention_mask_b"],
        scalars=device_batch["scalars"],
        labels=device_batch["labels"],
    )

    # Verify Outputs
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (Config.TRAIN_BATCH_SIZE, 3)
    assert outputs["loss"].item() > 0

    print("Forward pass verified. Loss calculated successfully.")

    # 4. Training Loop Execution
    print("\n>>> Starting Training Loop (1 Epoch)...")
    # This will train on the debug subset, validate, and save 'best_model.pth'
    run_training(train_loader, val_loader)

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # 5. Inference and Submission
    print("\n>>> Running Inference and Generating Submission...")
    # This uses TTA and generates submission.csv
    predict_and_submit(test_loader)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert all(
        col in sub_df.columns for col in required_cols
    ), "Submission columns incorrect"

    # Since we used debug mode, the submission might be truncated to the debug sample size
    # or padded depending on implementation. We just check it's not empty.
    assert len(sub_df) > 0, "Submission dataframe is empty"

    # Check probability sum (approximate due to float precision)
    row_sums = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    # TTA averaging might cause slight drift, but should be close to 1.0
    assert np.allclose(row_sums, 1.0, atol=1e-2), "Probabilities do not sum to 1.0"

    print(f"Submission verified. Shape: {sub_df.shape}")
    print(sub_df.head())

    # 6. Utility Verification
    print("\n>>> Verifying Metrics Utility...")
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    loss = compute_log_loss(y_true, y_pred)
    print(f"Computed Log Loss for dummy data: {loss:.4f}")
    assert loss < 0.3, "Log loss calculation seems incorrect for good predictions"

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
