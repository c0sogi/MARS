import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import process_data
from library.model import MORPE
from library.train_eval import run_training


def main():
    # 1. Setup and Configuration
    print("=== Setting up Configuration ===")
    seed_everything(Config.SEED)

    # Override Config to use a specific working directory for this demo
    # This ensures we don't interfere with other potential runs and allows fresh processing
    Config.CACHE_DIR = "./working/demo_execution/"
    Config.SUBMISSION_DIR = "./working/demo_execution/"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Verify Data Processing Logic
    print("\n=== Verifying Data Processing ===")
    # Call process_data. This handles loading, feature engineering (f_27 split), and encoding.
    df_train, df_val, df_test, meta_dict = process_data(load_cached_data=False)

    # Assertions to verify Feature Engineering
    print("Verifying feature engineering...")
    expected_char_cols = [f"char_{i}" for i in range(10)]
    for col in expected_char_cols:
        if col not in df_train.columns:
            raise AssertionError(
                f"Feature engineering failed: Column {col} missing from train data."
            )

    if "f_27" in df_train.columns:
        raise AssertionError(
            "Feature engineering failed: Original column 'f_27' should be dropped."
        )

    if "unique_char_count" not in df_train.columns:
        raise AssertionError(
            "Feature engineering failed: 'unique_char_count' feature missing."
        )

    # Assertions to verify Encoding and Scaling
    print("Verifying encoding and scaling...")
    cat_cols = meta_dict["cat_cols"]
    cont_cols = meta_dict["cont_cols"]

    # Categorical columns should be integers (encoded)
    if df_train[cat_cols[0]].dtype not in [np.int64, np.int32]:
        raise AssertionError(
            f"Encoding failed: Categorical column {cat_cols[0]} is not integer type."
        )

    # Continuous columns should be float (scaled)
    if df_train[cont_cols[0]].dtype not in [np.float32, np.float64]:
        raise AssertionError(
            f"Scaling failed: Continuous column {cont_cols[0]} is not float type."
        )

    print("Data processing verification passed.")

    # 3. Verify Model Architecture Logic
    print("\n=== Verifying Model Architecture ===")
    device = get_device()
    vocab_sizes_dict = meta_dict["vocab_sizes"]
    vocab_sizes = [vocab_sizes_dict[c] for c in cat_cols]

    # Instantiate model
    model = MORPE(
        vocab_sizes_list=vocab_sizes,
        num_cont=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.STREAMS,
    ).to(device)

    # Create dummy input batch
    batch_size = 4
    dummy_cat = torch.zeros((batch_size, len(cat_cols)), dtype=torch.long).to(device)
    dummy_cont = torch.randn((batch_size, len(cont_cols)), dtype=torch.float32).to(
        device
    )

    # Forward pass
    outputs = model(dummy_cat, dummy_cont)

    # Assertions for Model Output
    if not isinstance(outputs, list):
        raise AssertionError(
            "Model forward pass should return a list of outputs (one per stream)."
        )

    if len(outputs) != len(Config.STREAMS):
        raise AssertionError(
            f"Model returned {len(outputs)} streams, expected {len(Config.STREAMS)}."
        )

    for i, out in enumerate(outputs):
        if out.shape != (batch_size, 1):
            raise AssertionError(
                f"Stream {i} output shape mismatch. Expected ({batch_size}, 1), got {out.shape}"
            )

    print("Model architecture verification passed.")

    # 4. Run Training Pipeline (Fast Demonstration)
    print("\n=== Running Training Pipeline (Demo) ===")
    # We use a small sample size and few epochs to ensure quick execution
    # run_training handles dataset creation, loader setup, training loop, and prediction
    run_training(
        epochs=2,
        batch_size=1024,
        load_cached_data=True,  # Use the cache we just generated
        patience=1,
        sample_size=2000,  # Subset data for speed
    )

    # 5. Verify Submission
    print("\n=== Verifying Submission ===")
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_FILE}")

    submission_df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check shape - should match the full test set size (100,000 rows)
    # Note: Even though we trained on a subset, the inference runs on the full test set provided in metadata
    expected_test_len = len(df_test)
    if len(submission_df) != expected_test_len:
        raise AssertionError(
            f"Submission has {len(submission_df)} rows, expected {expected_test_len}."
        )

    # Check columns
    if list(submission_df.columns) != ["id", "target"]:
        raise AssertionError(
            f"Submission columns incorrect. Expected ['id', 'target'], got {list(submission_df.columns)}"
        )

    # Check value range
    if submission_df["target"].min() < 0 or submission_df["target"].max() > 1:
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    print(f"Submission verified successfully. Saved to {Config.SUBMISSION_FILE}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
