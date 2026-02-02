import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.dataset import EEGMultiModalDataset
from library.model import DualStreamEfficientNet
from library.engine import Engine


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    seed_everything(42)

    print("=== Starting Demonstration Script ===\n")

    # Override Config for a fast demonstration run
    print("Configuring for fast execution (Debug Mode)...")
    Config.DEBUG = True  # Uses only 100 samples for training/validation
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size for demonstration
    Config.OUTPUT_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.OUTPUT_DIR, "cache")
    Config.SUBMISSION_FILE = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Ensure output directories exist based on new config
    Config.setup()

    # -------------------------------------------------------------------------
    # 2. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[1/5] Verifying Dataset Logic...")

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Initialize dataset in train mode (returns targets)
    # We take a small slice of the dataframe manually for this specific check
    train_dataset = EEGMultiModalDataset(df_train.head(10), Config, mode="train")

    # Fetch one sample
    sample = train_dataset[0]
    eeg_spec = sample["eeg_spec"]
    kaggle_spec = sample["kaggle_spec"]
    target = sample["target"]

    print(f"  Sample 0 - EEG Spec Shape: {eeg_spec.shape}")
    print(f"  Sample 0 - Kaggle Spec Shape: {kaggle_spec.shape}")
    print(f"  Sample 0 - Target: {target}")

    # Assertions
    # Expected shape: (3, 512, 512) -> 3 channels (repeated), 512x512 image
    expected_shape = (3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        eeg_spec.shape == expected_shape
    ), f"EEG spec shape mismatch: {eeg_spec.shape}"
    assert (
        kaggle_spec.shape == expected_shape
    ), f"Kaggle spec shape mismatch: {kaggle_spec.shape}"
    assert target.shape == (6,), f"Target shape mismatch: {target.shape}"

    # Verify target is a valid probability distribution (sums to ~1)
    assert torch.isclose(
        target.sum(), torch.tensor(1.0), atol=1e-5
    ), f"Target probabilities do not sum to 1: {target.sum()}"

    print("  ✓ Dataset logic verified.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Model Architecture...")

    model = DualStreamEfficientNet(Config)
    model.eval()

    # Create dummy input tensors
    batch_size = 2
    dummy_eeg = torch.randn(batch_size, *expected_shape)
    dummy_kaggle = torch.randn(batch_size, *expected_shape)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_eeg, dummy_kaggle)

    print(f"  Input Batch Size: {batch_size}")
    print(f"  Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected ({batch_size}, {Config.NUM_CLASSES}), got {logits.shape}"

    print("  ✓ Model architecture verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification (Engine)
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Training Loop (Engine)...")

    engine = Engine(Config)

    # This will run for 1 epoch on 100 samples (due to Config.DEBUG=True)
    print("  Starting training (this may take a moment)...")
    best_model_path = engine.run_training()

    # Assertions
    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"  ✓ Training completed. Model saved to: {best_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying Inference and Submission Generation...")

    # Generate submission using the model just trained
    # Note: This runs on the full test set defined in test.csv
    engine.generate_submission(best_model_path)

    # Assertions
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"  Submission Shape: {submission_df.shape}")

    # Check required columns
    expected_cols = ["eeg_id"] + Config.CLASS_NAMES
    missing_cols = [col for col in expected_cols if col not in submission_df.columns]
    assert not missing_cols, f"Submission missing columns: {missing_cols}"

    # Check that probabilities sum to ~1 for the first row
    row_sum = submission_df[Config.CLASS_NAMES].iloc[0].sum()
    assert np.isclose(
        row_sum, 1.0, atol=1e-4
    ), f"Submission row probabilities do not sum to 1: {row_sum}"

    print("  ✓ Inference verified.")

    # -------------------------------------------------------------------------
    # 6. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Metric Calculation...")

    # Synthetic ground truth (one-hot and soft labels)
    y_true = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]])

    # Synthetic predictions (slightly off)
    y_pred = np.array(
        [[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]]
    )

    score = kl_divergence_score(y_true, y_pred)
    print(f"  Calculated KL Score: {score:.6f}")

    # Assertions
    assert score >= 0, "KL Divergence cannot be negative."
    assert isinstance(score, float), "Metric should return a float."

    print("  ✓ Metric verified.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
