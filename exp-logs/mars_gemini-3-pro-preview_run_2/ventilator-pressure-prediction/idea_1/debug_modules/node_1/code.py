import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data, get_data_loaders
from library.model import BiLSTMRegressor
from library.trainer import Trainer


def main():
    print("=== Ventilator Pressure Prediction: Pipeline Demonstration ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config settings to run a fast, small-scale test
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Use only 200 breaths for train/val/test
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 32  # Smaller batch size for demo

    # Use a temporary directory for this demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_CHECKPOINT = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Create necessary directories
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: DEBUG=True, EPOCHS=2, Sample=200 breaths.")

    # ------------------------------------------------------------------------
    # 2. Data Preparation
    # ------------------------------------------------------------------------
    print("\n[2] Processing Data (prepare_data)...")

    # load_cached_data=False forces the raw CSV processing logic to run
    X_train, y_train, X_val, y_val, X_test, test_ids = prepare_data(
        load_cached_data=False
    )

    print(f"    Train X shape: {X_train.shape}")
    print(f"    Train y shape: {y_train.shape}")

    # Validation
    assert X_train.ndim == 3, "X_train must be 3D (Num_Breaths, Seq_Len, Features)"
    assert (
        X_train.shape[1] == Config.SEQ_LEN
    ), f"Sequence length must be {Config.SEQ_LEN}"
    assert (
        X_train.shape[2] == Config.INPUT_DIM
    ), f"Input dimension must be {Config.INPUT_DIM}"
    assert y_train.shape == (X_train.shape[0], Config.SEQ_LEN), "Target shape mismatch"
    print("    Data preparation verified.")

    # ------------------------------------------------------------------------
    # 3. Data Loaders
    # ------------------------------------------------------------------------
    print("\n[3] Initializing Data Loaders (get_data_loaders)...")

    train_loader, val_loader, test_loader, t_ids = get_data_loaders(
        load_cached_data=True
    )

    # Validate one batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"    Batch X shape: {x_batch.shape}")
    print(f"    Batch y shape: {y_batch.shape}")

    assert x_batch.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    assert y_batch.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    print("    Data loaders verified.")

    # ------------------------------------------------------------------------
    # 4. Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Initializing Model (BiLSTMRegressor)...")

    model = BiLSTMRegressor()
    model.to(Config.DEVICE)

    # Perform a dummy forward pass to verify architecture
    with torch.no_grad():
        dummy_input = torch.randn(
            Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM
        ).to(Config.DEVICE)
        output = model(dummy_input)

    print(f"    Model Output shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Model output shape mismatch"
    print("    Model architecture verified.")

    # ------------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Trainer.fit)...")

    trainer = Trainer(model, train_loader, val_loader, test_loader, t_ids)
    trainer.fit()

    # Verify that the best model was saved
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created."
    print(f"    Training complete. Checkpoint saved to {Config.MODEL_CHECKPOINT}")

    # ------------------------------------------------------------------------
    # 6. Inference
    # ------------------------------------------------------------------------
    print("\n[6] Generating Predictions (Trainer.predict)...")

    trainer.predict()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {df_sub.shape}")

    # Validate submission format
    assert list(df_sub.columns) == [
        "id",
        "pressure",
    ], "Submission columns are incorrect."
    assert not df_sub.isnull().values.any(), "Submission contains NaNs."

    # In DEBUG mode, we expect predictions for the subset of test breaths
    # Each breath has 80 time steps.
    expected_rows = len(t_ids)
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print(f"    Inference verified. Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
