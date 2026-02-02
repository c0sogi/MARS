import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_normalized_levenshtein
from library.data_loader import get_dataloaders
from library.model import HNGKN
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("=== Starting Demonstration ===")

    # ==========================================
    # 1. Setup Configuration for Demo
    # ==========================================
    print("\n[1] Configuring environment for fast execution...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10  # Use only 10 samples per split
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set temporary directories for this run
    Config.CACHE_DIR = "./working/demo_execution/cache"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.MODEL_SAVE_PATH = "./working/demo_execution/best_model.pth"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up any previous demo artifacts
    if os.path.exists("./working/demo_execution"):
        shutil.rmtree("./working/demo_execution")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration set. Debug mode enabled.")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n[2] Verifying utility functions...")

    # Test Normalized Levenshtein Distance
    # Case 1: Perfect match
    score_perfect = compute_normalized_levenshtein([[1, 2, 3]], [[1, 2, 3]])
    assert score_perfect == 0.0, f"Expected 0.0, got {score_perfect}"

    # Case 2: One insertion (Distance=1, GT Length=3) -> 1/3
    score_mismatch = compute_normalized_levenshtein([[1, 2]], [[1, 2, 3]])
    expected_score = 1.0 / 3.0
    assert (
        abs(score_mismatch - expected_score) < 1e-6
    ), f"Expected {expected_score}, got {score_mismatch}"

    print("Utilities verified successfully.")

    # ==========================================
    # 3. Data Loading Demonstration
    # ==========================================
    print("\n[3] Testing Data Loading...")

    # Force re-computation of cache for the debug subset by setting load_cached_data=False
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=False
    )

    # Fetch one batch to verify shapes
    features, labels = next(iter(train_loader))

    print(
        f"Batch Features Shape: {features.shape}"
    )  # Expected: (Batch, Window, InputDim)
    print(f"Batch Labels Shape: {labels.shape}")  # Expected: (Batch, Window)

    # Assertions
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
        Config.INPUT_DIM,
    ), f"Feature shape mismatch. Expected {(Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM)}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE, Config.WINDOW_SIZE)}"

    print("Data Loaders functional and shapes correct.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[4] Testing Model Architecture...")

    model = HNGKN()
    model.eval()

    # Create dummy input matching the batch shape
    dummy_input = torch.randn(Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM)

    # Forward pass
    with torch.no_grad():
        logits1, logits2, logits3 = model(dummy_input)

    # Verify outputs (Deep Supervision returns 3 sets of logits)
    expected_output_shape = (Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NUM_CLASSES)

    assert (
        logits1.shape == expected_output_shape
    ), f"Stage 1 logits shape mismatch: {logits1.shape}"
    assert (
        logits2.shape == expected_output_shape
    ), f"Stage 2 logits shape mismatch: {logits2.shape}"
    assert (
        logits3.shape == expected_output_shape
    ), f"Stage 3 logits shape mismatch: {logits3.shape}"

    print("Model forward pass successful. Output shapes correct.")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n[5] Executing Training Loop (2 Epochs)...")

    # Initialize Trainer
    trainer = Trainer(Config)

    # Run training
    trainer.train()

    # Verify model checkpoint creation
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint was not created at {Config.MODEL_SAVE_PATH}"
        )

    print("Training completed and model saved.")

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\n[6] Running Inference and Generating Submission...")

    # Run inference using the standalone function (which uses the saved model)
    # We use load_cached_data=True to reuse the cache generated in step 3
    run_inference(load_cached_data=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Check content of submission
    df_sub = pd.read_csv(Config.SUBMISSION_PATH, header=None)
    print(f"Submission file created with {len(df_sub)} rows.")

    # In debug mode with 10 samples, we expect 10 rows
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} predictions, found {len(df_sub)}"

    # Check format of first row (SessionID, ...)
    first_row = df_sub.iloc[0, 0]
    assert (
        isinstance(first_row, str) and "Sample" in first_row
    ), f"Unexpected submission format. First column value: {first_row}"

    print("Inference pipeline verified successfully.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
