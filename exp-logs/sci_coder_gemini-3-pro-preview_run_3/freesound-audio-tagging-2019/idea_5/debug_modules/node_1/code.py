import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_per_class_lwlrap,
    calculate_overall_lwlrap,
    mixup_data,
    mixup_criterion,
)
from library.model import AudioEfficientNet
from library.trainer import Trainer


def setup_demo_config():
    """
    Overrides default configuration for a fast demonstration run.
    """
    print("[Demo] Setting up configuration for fast execution...")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small number for speed

    # Training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure we use a clean working directory for this demo if needed
    # (The library uses ./working/idea_5, we leave it as is but ensure cache doesn't conflict)
    # We will handle cache via the load_cached_data=False flag in trainer.

    # Set device to CPU if GPU is not critical for this tiny test,
    # though the library auto-detects. We'll stick to library detection.
    print(f"[Demo] Device: {Config.DEVICE}")
    print(f"[Demo] Debug Mode: {Config.DEBUG}")
    print(f"[Demo] Epochs: {Config.EPOCHS}")


def verify_utils():
    """
    Verifies the logic of utility functions, specifically the metric calculation.
    """
    print("\n[Demo] Verifying Utility Functions...")

    # 1. Verify LWLRAP (Label-Weighted Label-Ranking Average Precision)
    # Scenario: 2 samples, 3 classes
    # Sample 0: Truth [1, 0, 0], Pred [0.8, 0.1, 0.1] -> Rank 1 (Correct) -> Precision 1.0
    # Sample 1: Truth [0, 1, 0], Pred [0.2, 0.5, 0.3] -> Rank 1 (Correct) -> Precision 1.0

    truth = np.array([[1, 0, 0], [0, 1, 0]])
    scores = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])

    score = calculate_overall_lwlrap(truth, scores)
    assert np.isclose(score, 1.0), f"Expected score 1.0, got {score}"
    print("  - LWLRAP Perfect Score Check: Passed")

    # Scenario: Imperfect ranking
    # Sample 0: Truth [1, 0], Pred [0.4, 0.6]
    #   Rank of class 0 is 2. Relevant items at rank 2 = 1. Precision = 1/2 = 0.5.
    truth = np.array([[1, 0]])
    scores = np.array([[0.4, 0.6]])
    score = calculate_overall_lwlrap(truth, scores)
    assert np.isclose(score, 0.5), f"Expected score 0.5, got {score}"
    print("  - LWLRAP Imperfect Score Check: Passed")

    # 2. Verify Mixup
    # Create dummy tensors
    batch_size = 4
    channels = 1
    freq = 128
    time_steps = 100
    x = torch.randn(batch_size, channels, freq, time_steps)
    y = torch.randint(0, 2, (batch_size, 10)).float()  # Multi-label binary targets

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    assert mixed_x.shape == x.shape, "Mixup output shape mismatch"
    assert y_a.shape == y.shape, "Mixup target A shape mismatch"
    assert y_b.shape == y.shape, "Mixup target B shape mismatch"
    assert 0 <= lam <= 1, "Lambda out of range"
    print("  - Mixup Logic Check: Passed")


def verify_model():
    """
    Verifies the model architecture and forward pass.
    """
    print("\n[Demo] Verifying Model Architecture...")

    model = AudioEfficientNet()
    model.eval()

    # Create a dummy input tensor matching the expected input dimensions
    # Shape: (Batch, 1, Freq, Time)
    # Based on Config: N_MELS=128. Time depends on duration, but model handles variable length (global pooling).
    dummy_input = torch.randn(2, 1, Config.N_MELS, 500)

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape
    expected_shape = (2, Config.NUM_CLASSES)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"  - Model Forward Pass Check: Passed (Output Shape: {output.shape})")


def run_pipeline():
    """
    Runs the full training and prediction pipeline using the Trainer class.
    """
    print("\n[Demo] Running Training Pipeline...")

    # Initialize Trainer
    trainer = Trainer()

    # Run Fit
    # Important: Set load_cached_data=False to ensure we generate the small debug dataset
    # instead of loading a potentially large cached dataset from disk.
    test_loader = trainer.fit(load_cached_data=False)

    print("[Demo] Training complete. Running prediction...")

    # Run Predict
    trainer.predict(test_loader)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"[Demo] Submission file generated at {Config.SUBMISSION_PATH}")
        print(f"  - Shape: {df.shape}")

        # Verify rows match debug sample size (or close to it depending on test set size in debug)
        # In debug mode, test set is also sliced to DEBUG_SAMPLE_SIZE
        assert (
            len(df) == Config.DEBUG_SAMPLE_SIZE
        ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df)}"

        # Verify columns (fname + 80 classes)
        assert (
            len(df.columns) == Config.NUM_CLASSES + 1
        ), f"Submission column count mismatch. Expected {Config.NUM_CLASSES + 1}, got {len(df.columns)}"

        print("  - Submission Format Check: Passed")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # 1. Set Seed for Reproducibility
    seed_everything(Config.SEED)

    # 2. Setup Config for Speed
    setup_demo_config()

    # 3. Verify Components
    verify_utils()
    verify_model()

    # 4. Run Full Pipeline
    run_pipeline()

    print("\n[Demo] All demonstration steps completed successfully.")
