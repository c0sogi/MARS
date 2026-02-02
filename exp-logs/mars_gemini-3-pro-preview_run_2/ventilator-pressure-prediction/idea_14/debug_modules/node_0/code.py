import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_data_loaders
from library.model import DFL_GI_BiLSTM
from library.train import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Use only 200 breaths for the demo
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 32  # Smaller batch size for the small subset

    # Redirect working directory to a clean demo folder
    Config.WORKING_DIR = "./working/demo_execution_script"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # IMPORTANT: Re-bind dependent paths in Config since they were initialized at import time
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.SCALER_PARAMS_PATH = os.path.join(Config.WORKING_DIR, "scaler_params.npz")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Force reload to ensure we process the debug subset and save to new cache
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    # Fetch a single batch to verify shapes
    try:
        inputs, targets, u_out = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty! Check data loading logic.")

    print(
        f"Batch shapes -> Inputs: {inputs.shape}, Targets: {targets.shape}, u_out: {u_out.shape}"
    )

    # Assertions
    # Shape: (Batch, Seq_Len, Features)
    assert inputs.ndim == 3, "Inputs should be 3-dimensional (Batch, Seq, Feat)"
    assert (
        inputs.shape[1] == Config.SEQ_LEN
    ), f"Sequence length should be {Config.SEQ_LEN}"
    assert (
        inputs.shape[2] == Config.INPUT_DIM
    ), f"Input dim should be {Config.INPUT_DIM}"

    # Targets: (Batch, Seq_Len)
    assert targets.ndim == 2, "Targets should be 2-dimensional (Batch, Seq)"
    assert targets.shape[1] == Config.SEQ_LEN, "Target sequence length mismatch"

    # u_out: (Batch, Seq_Len)
    assert u_out.ndim == 2, "u_out should be 2-dimensional"

    print("Data Pipeline verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = DFL_GI_BiLSTM().to(device)

    # Move batch to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(inputs)

    print(f"Model Output Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        inputs.shape[0],
        Config.SEQ_LEN,
    ), f"Model output shape mismatch. Expected {(inputs.shape[0], Config.SEQ_LEN)}, got {preds.shape}"

    print("Model verification passed.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Execution
    # ------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (2 Epochs)...")

    trainer = Trainer()

    # Run training
    # This uses the modified Config.EPOCHS = 2
    trainer.fit(train_loader, val_loader)

    # Verify artifacts
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_PATH} after training."
        )

    print("Training complete. Checkpoint saved.")

    # ------------------------------------------------------------------------
    # 5. Inference Execution
    # ------------------------------------------------------------------------
    print("\n[5] Executing Inference Pipeline...")

    # This function loads the best model from Config.MODEL_PATH and generates submission
    generate_submission()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Assertions for submission
    expected_rows = Config.DEBUG_SAMPLE_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    assert list(df_sub.columns) == [
        "id",
        "pressure",
    ], f"Submission columns mismatch. Expected ['id', 'pressure'], got {list(df_sub.columns)}"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("Inference verification passed.")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
