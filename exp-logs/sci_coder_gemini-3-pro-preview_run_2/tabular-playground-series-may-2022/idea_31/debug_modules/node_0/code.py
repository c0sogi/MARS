import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import process_data, ManufacturingDataset
from library.model import PostNormHybridSwiGLU
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides
    # We modify the Config class attributes directly to adapt for a quick demo run.
    print("\n[Step 1] Configuring environment...")
    seed_everything(42)

    # Override Config for speed and isolation
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_PATH = os.path.join(Config.WORKING_DIR, "processed_data_demo.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Configuration set: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Processing Demonstration
    print("\n[Step 2] Demonstrating Data Processing...")

    # Force reprocessing to ensure we test the logic (load_cached_data=False)
    data = process_data(Config, load_cached_data=False)

    (
        X_train_seq,
        X_train_cont,
        y_train,
        X_val_seq,
        X_val_cont,
        y_val,
        X_test_seq,
        X_test_cont,
        test_ids,
    ) = data

    # Assertions to verify data shapes
    print("Verifying data shapes...")
    assert X_train_seq.ndim == 2, "Sequence data should be 2D"
    assert (
        X_train_seq.shape[1] == Config.SEQ_LEN
    ), f"Sequence length should be {Config.SEQ_LEN}"
    assert X_train_cont.ndim == 2, "Continuous data should be 2D"
    assert (
        X_train_cont.shape[1] == Config.NUM_FEATURES
    ), f"Num features should be {Config.NUM_FEATURES}"
    assert len(X_train_seq) == len(
        y_train
    ), "Mismatch between training features and targets"

    print(f"Train samples: {len(X_train_seq)}")
    print(f"Val samples:   {len(X_val_seq)}")
    print(f"Test samples:  {len(X_test_seq)}")

    # Verify Dataset Class
    print("Verifying ManufacturingDataset class...")
    ds = ManufacturingDataset(X_train_seq[:10], X_train_cont[:10], y_train[:10])
    assert len(ds) == 10

    # Check item retrieval
    seq_sample, cont_sample, target_sample = ds[0]
    assert torch.is_tensor(seq_sample)
    assert torch.is_tensor(cont_sample)
    assert torch.is_tensor(target_sample)
    assert seq_sample.dtype == torch.long
    assert cont_sample.dtype == torch.float32

    print("Data processing and Dataset class verified successfully.")

    # 3. Model Architecture Demonstration
    print("\n[Step 3] Demonstrating Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PostNormHybridSwiGLU(Config).to(device)

    # Create dummy batch
    batch_size_demo = 4
    dummy_seq = torch.randint(
        0, Config.VOCAB_SIZE, (batch_size_demo, Config.SEQ_LEN)
    ).to(device)
    dummy_cont = torch.randn(batch_size_demo, Config.NUM_FEATURES).to(device)

    # Test Training Mode (Multi-Sample Dropout)
    model.train()
    out_train = model(dummy_seq, dummy_cont)
    # Expected shape: [Heads, Batch]
    assert out_train.shape == (
        Config.MSD_HEADS,
        batch_size_demo,
    ), f"Training output shape mismatch. Expected ({Config.MSD_HEADS}, {batch_size_demo}), got {out_train.shape}"

    # Test Eval Mode (Inference)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_seq, dummy_cont)
    # Expected shape: [Batch]
    assert out_eval.shape == (
        batch_size_demo,
    ), f"Eval output shape mismatch. Expected ({batch_size_demo},), got {out_eval.shape}"

    print("Model forward pass (Train/Eval) verified successfully.")

    # 4. Full Training Pipeline Execution
    print("\n[Step 4] Executing Training Pipeline (Debug Mode)...")

    # Run training with debug_mode=True to use a subset of data (2048 samples)
    # This uses the modified Config settings (2 epochs)
    run_training(debug_mode=True)

    print("Training pipeline execution completed.")

    # 5. Output Verification
    print("\n[Step 5] Verifying Outputs...")

    # Check if submission file exists
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Check if model file exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns"

    # Check ID alignment with test set
    # Note: run_training(debug_mode=True) only slices train/val, test set remains full size (100000)
    # The submission should match the full test set size.
    expected_test_size = 100000  # From metadata description
    assert (
        len(df_sub) == expected_test_size
    ), f"Submission row count mismatch. Expected {expected_test_size}, got {len(df_sub)}"

    # Check probability range
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("Output verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
