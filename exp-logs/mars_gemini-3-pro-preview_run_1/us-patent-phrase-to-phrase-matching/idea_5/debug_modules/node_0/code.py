import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_cpc_texts, PearsonDataset
from library.engine import run_fold, predict_and_submit


def main():
    print("Initializing Demonstration Script...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for speed and demonstration purposes
    Config.working_dir = "./working/demo_run"
    Config.models_dir = os.path.join(Config.working_dir, "models")
    Config.predictions_dir = os.path.join(Config.working_dir, "predictions")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Reduce computational load
    Config.epochs = 1
    Config.num_folds = 1  # Run only one fold
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.num_workers = 0  # Disable multiprocessing for small data to avoid overhead

    # Create necessary directories
    Config.create_dirs()

    # Set seed for reproducibility
    seed_everything(Config.seed)

    device = Config.device
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Preparation
    # =========================================================================
    print("Loading and preprocessing data...")

    # Load metadata
    if not os.path.exists(Config.train_path):
        raise FileNotFoundError(f"Train metadata not found at {Config.train_path}")

    df_train_full = pd.read_csv(Config.train_path)
    df_test_full = pd.read_csv(Config.test_path)

    # Subset data for speed (32 train samples, 16 test samples)
    df_train_demo = df_train_full.head(32).copy()
    df_test_demo = df_test_full.head(16).copy()

    # Load CPC Contexts
    cpc_texts = get_cpc_texts(load_cached_data=True)

    # Map context codes to full text
    # The dataset class expects 'context_text' or handles raw 'context'
    # We map it here to demonstrate best practice as per dataset.py hints
    df_train_demo["context_text"] = df_train_demo["context"].map(cpc_texts)
    df_test_demo["context_text"] = df_test_demo["context"].map(cpc_texts)

    # Fill missing contexts if any (fallback to code)
    df_train_demo["context_text"] = df_train_demo["context_text"].fillna(
        df_train_demo["context"]
    )
    df_test_demo["context_text"] = df_test_demo["context_text"].fillna(
        df_test_demo["context"]
    )

    # =========================================================================
    # 3. Tokenizer & Datasets
    # =========================================================================
    print(f"Initializing Tokenizer ({Config.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Split train demo into Train/Validation
    # We use first 24 for train, last 8 for validation
    train_subset = df_train_demo.iloc[:24].reset_index(drop=True)
    val_subset = df_train_demo.iloc[24:].reset_index(drop=True)

    print("Creating Datasets...")
    train_dataset = PearsonDataset(train_subset, tokenizer, Config.max_len)
    val_dataset = PearsonDataset(val_subset, tokenizer, Config.max_len)
    test_dataset = PearsonDataset(df_test_demo, tokenizer, Config.max_len, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True, drop_last=True
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )

    # =========================================================================
    # 4. Training Loop (Fold 0)
    # =========================================================================
    print("Starting Training for Fold 0...")

    # run_fold executes the training loop and saves the best model
    best_score = run_fold(0, train_loader, valid_loader, device)

    print(f"Fold 0 Training Complete. Best Validation Score: {best_score:.4f}")

    # Verify model file was created
    expected_model_path = os.path.join(Config.models_dir, "model_fold_0.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {expected_model_path}")

    # =========================================================================
    # 5. Inference & Submission
    # =========================================================================
    print("Generating Submission...")

    # predict_and_submit reads the test file from Config.test_path to get IDs.
    # Since we are using a subset, we must point Config.test_path to a file containing only our subset.
    temp_test_path = os.path.join(Config.working_dir, "temp_test_subset.csv")
    df_test_demo.to_csv(temp_test_path, index=False)

    # Temporarily override test_path in Config
    original_test_path = Config.test_path
    Config.test_path = temp_test_path

    try:
        # Run inference using the trained model(s)
        predict_and_submit(test_loader, device)
    finally:
        # Restore path just in case
        Config.test_path = original_test_path

    # =========================================================================
    # 6. Verification
    # =========================================================================
    print("Verifying Submission...")

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    df_sub = pd.read_csv(Config.submission_path)

    # Check 1: Length
    if len(df_sub) != len(df_test_demo):
        raise AssertionError(
            f"Submission length mismatch. Expected {len(df_test_demo)}, got {len(df_sub)}"
        )

    # Check 2: Columns
    required_cols = {"id", "score"}
    if not required_cols.issubset(df_sub.columns):
        raise AssertionError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Check 3: Data Types
    if not pd.api.types.is_numeric_dtype(df_sub["score"]):
        raise AssertionError("Score column is not numeric.")

    print("\n" + "=" * 40)
    print("SUCCESS: Pipeline demonstration completed.")
    print(f"Submission saved to: {Config.submission_path}")
    print("Top 3 Predictions:")
    print(df_sub.head(3))
    print("=" * 40)


if __name__ == "__main__":
    main()
