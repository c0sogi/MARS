import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.train_eval import run_training
from library.data_utils import preprocess_pipeline


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    print("Initializing Demo Execution...")
    set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Monkey-Patch Configuration for Speed and Isolation
    # -------------------------------------------------------------------------
    # We override the global configuration to run a minimal version of the task.
    # This ensures the script completes quickly while exercising all code paths.

    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")

    # Clean up previous demo run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up configuration. Working directory: {demo_dir}")

    # Override paths to keep demo artifacts separate
    Config.IDEA_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override cache paths so we don't accidentally use (or overwrite)
    # the main experiment's processed data during this demo.
    Config.CACHE_TRAIN = os.path.join(demo_dir, "train_processed.parquet")
    Config.CACHE_VAL = os.path.join(demo_dir, "val_processed.parquet")
    Config.CACHE_TEST = os.path.join(demo_dir, "test_processed.parquet")
    Config.CACHE_METADATA = os.path.join(demo_dir, "metadata.npy")

    # Override Hyperparameters for Speed
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4096  # Large batch size to speed up iteration on GPU

    # Check Device
    print(f"Running on device: {Config.DEVICE}")
    if Config.DEVICE == "cpu":
        print("Warning: Running on CPU. This might be slow.")

    # -------------------------------------------------------------------------
    # 2. Execute Training Pipeline
    # -------------------------------------------------------------------------
    # run_training handles:
    # - Data loading (and preprocessing if cache is missing)
    # - Model instantiation (SDPEModel)
    # - Training loop
    # - Validation
    # - Prediction generation

    print("\nStarting Training Pipeline via library.train_eval.run_training...")
    try:
        # load_cached_data=False forces the preprocessing pipeline to run,
        # verifying the feature engineering logic in data_utils.py.
        run_training(load_cached_data=False)
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 3. Validation and Assertions
    # -------------------------------------------------------------------------
    print("\nVerifying outputs...")

    # A. Check Model Artifact
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
    print(
        f"[PASS] Model file generated: {os.path.getsize(Config.MODEL_PATH) / 1024 / 1024:.2f} MB"
    )

    # B. Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # C. Validate Submission Format
    required_columns = ["id", "target"]
    if list(df_sub.columns) != required_columns:
        raise AssertionError(
            f"Submission columns mismatch. Expected {required_columns}, got {list(df_sub.columns)}"
        )

    # D. Validate Submission Shape (Test set is 100,000 rows)
    expected_shape = (100000, 2)
    if df_sub.shape != expected_shape:
        raise AssertionError(
            f"Submission shape mismatch. Expected {expected_shape}, got {df_sub.shape}"
        )

    # E. Validate ID correctness (Range 900000 - 999999 based on typical test sets,
    # but strictly checking against metadata/test.csv ids is safer).
    # Here we just check they are integers and unique.
    if not pd.api.types.is_integer_dtype(df_sub["id"]):
        raise TypeError("Column 'id' should be of integer type.")

    if df_sub["id"].nunique() != len(df_sub):
        raise AssertionError("Duplicate IDs found in submission.")

    # F. Validate Probabilities
    probs = df_sub["target"]
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError("Target probabilities are out of bounds [0, 1].")

    if probs.isnull().any():
        raise ValueError("NaN values found in target predictions.")

    print(f"[PASS] Submission file valid. Shape: {df_sub.shape}")
    print(f"      Mean Prediction: {probs.mean():.4f}")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
