import os
import pandas as pd
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.trainer import run_training


def main():
    print("Initializing demo execution...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We modify the global Config class attributes directly to redirect outputs
    # and adjust hyperparameters for a fast demonstration run.

    # Set output directories to a demo specific folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Ensure these directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update file paths to point to the new working directory
    # The data loader handles the .pt -> .npz extension switch internally
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_data.pt")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_data.pt")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_data.pt")
    Config.SCALER_CACHE_PATH = os.path.join(Config.WORKING_DIR, "scalers.pt")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce hyperparameters for speed
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.ATOMIC_HIDDEN_DIM = 64  # Smaller model width
    Config.GLOBAL_HIDDEN_DIM = 32
    Config.FUSION_HIDDEN_DIM = 64

    # Use a small subset of data (e.g., 50 samples) for processing
    DEBUG_SIZE = 50

    print(f"Configuration updated.")
    print(f"  Working Dir: {Config.WORKING_DIR}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Debug Sample Size: {DEBUG_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Run Training Pipeline
    # -------------------------------------------------------------------------
    # This function handles:
    # - Data loading (and caching if not present)
    # - Model instantiation
    # - Training loop with validation
    # - Saving the best model
    # - Generating predictions on the test set
    print("\nStarting training pipeline...")
    run_training(config=Config, debug_sample_size=DEBUG_SIZE)

    # -------------------------------------------------------------------------
    # 3. Verification
    # -------------------------------------------------------------------------
    print("\nVerifying outputs...")

    # Verify Model Checkpoint
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"[PASS] Model checkpoint found at: {Config.MODEL_SAVE_PATH}")
        # Verify we can load it
        try:
            state_dict = torch.load(
                Config.MODEL_SAVE_PATH, map_location=torch.device("cpu")
            )
            print(
                f"[PASS] Model state dictionary loaded successfully. Keys: {len(state_dict)}"
            )
        except Exception as e:
            raise AssertionError(f"Failed to load model checkpoint: {e}")
    else:
        raise FileNotFoundError(
            f"Model checkpoint not created at {Config.MODEL_SAVE_PATH}"
        )

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"[PASS] Submission file found at: {Config.SUBMISSION_PATH}")

        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission shape: {df.shape}")

        # Check columns
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        if list(df.columns) != expected_cols:
            raise AssertionError(
                f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"
            )

        # Check row count (should match DEBUG_SIZE since test set is also truncated)
        if len(df) != DEBUG_SIZE:
            raise AssertionError(
                f"Expected {DEBUG_SIZE} rows in submission, found {len(df)}"
            )

        # Check for NaN/Inf
        if df.isnull().values.any():
            raise AssertionError("Submission contains NaN values.")

        print("[PASS] Submission content verified.")
    else:
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
