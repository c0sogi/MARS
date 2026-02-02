import os
import torch
import numpy as np
import pandas as pd
import shutil
import time

# Import from the provided library
from library.config import WORKING_DIR, SUBMISSION_DIR, DEVICE, set_seed, SEED
from library.utils import rmse_score
from library.dataset import get_kfold_loaders, get_test_loader
from library.model import ResidualShallowUNet
from library.train import run_training
from library.inference import generate_submission, predict_tta


def test_utils():
    """Verifies utility functions."""
    print("Testing Utilities...")

    # Test RMSE Score
    y_true = [1.0, 0.0, 0.5]
    y_pred = [1.0, 0.0, 0.5]
    score = rmse_score(y_true, y_pred)
    assert score == 0.0, f"RMSE should be 0.0 for identical arrays, got {score}"

    y_true = [0.0, 1.0]
    y_pred = [1.0, 0.0]
    # MSE = ((0-1)^2 + (1-0)^2) / 2 = 1. RMSE = 1.
    score = rmse_score(y_true, y_pred)
    assert np.isclose(score, 1.0), f"RMSE should be 1.0, got {score}"

    print("Utilities verified.\n")


def test_dataset_loading():
    """Verifies dataset loading and augmentation pipeline."""
    print("Testing Dataset Loading...")

    # Get loaders for 2 folds
    n_folds = 2
    loaders = get_kfold_loaders(n_folds=n_folds, load_cached_data=False)

    assert (
        len(loaders) == n_folds
    ), f"Expected {n_folds} sets of loaders, got {len(loaders)}"

    train_loader, val_loader = loaders[0]

    # Check Train Batch
    # Train loader uses batch_size=16 (from config) and random crops to IMG_SIZE=160
    try:
        noisy, residual = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    print(f"Train batch shape: {noisy.shape}")
    assert len(noisy.shape) == 4, "Train noisy images should be 4D (B, C, H, W)"
    assert noisy.shape[1] == 1, "Train images should have 1 channel"
    assert (
        noisy.shape[2] == 160 and noisy.shape[3] == 160
    ), "Train images should be cropped to 160x160"
    assert noisy.shape == residual.shape, "Noisy and Residual shapes must match"

    # Verify value ranges
    assert (
        noisy.max() <= 1.0 and noisy.min() >= 0.0
    ), "Noisy images should be normalized to [0, 1]"
    # Residual can be negative (Noisy - Clean), range approx [-1, 1]
    assert (
        residual.max() <= 1.0 and residual.min() >= -1.0
    ), "Residuals should be in range [-1, 1]"

    # Check Validation Batch
    # Val loader uses batch_size=1 and original image size
    try:
        val_noisy, val_residual = next(iter(val_loader))
    except StopIteration:
        raise AssertionError("Val loader is empty.")

    print(f"Val batch shape: {val_noisy.shape}")
    assert val_noisy.shape[0] == 1, "Val batch size should be 1"
    # Dimensions vary, just check rank
    assert len(val_noisy.shape) == 4

    print("Dataset loading verified.\n")


def test_model_architecture():
    """Verifies the U-Net model architecture."""
    print("Testing Model Architecture...")

    model = ResidualShallowUNet(n_channels=1, n_classes=1).to(DEVICE)
    model.eval()

    # Create dummy input: Batch=2, Channel=1, Size=160x160
    dummy_input = torch.randn(2, 1, 160, 160).to(DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert (
        output.shape == dummy_input.shape
    ), f"Output shape {output.shape} does not match input shape {dummy_input.shape}"

    # Test TTA function
    tta_output = predict_tta(model, dummy_input, DEVICE)
    assert tta_output.shape == dummy_input.shape, "TTA output shape mismatch"

    print("Model architecture verified.\n")


def test_training_process():
    """Runs a short training loop to verify the training pipeline."""
    print("Testing Training Process...")

    # Ensure clean working directory for this run
    if os.path.exists(WORKING_DIR):
        # Don't delete, just ensure we can write to it.
        # The library code creates it if missing.
        pass

    # Run training for 2 folds, 1 epoch each to be fast
    # This verifies: Model init, Optimizer, Loss calculation, Backprop, Validation, Checkpointing
    run_training(epochs=1, n_folds=2, load_cached_data=True)

    # Verify model files were created
    for fold in range(2):
        model_path = os.path.join(WORKING_DIR, f"model_fold_{fold}.pth")
        assert os.path.exists(
            model_path
        ), f"Model checkpoint for fold {fold} not found at {model_path}"

    print("Training process verified.\n")


def test_inference_and_submission():
    """Runs the inference pipeline and generates a submission file."""
    print("Testing Inference and Submission...")

    # Ensure submission directory exists (handled by config, but double check)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Remove existing submission if any
    if os.path.exists(submission_path):
        os.remove(submission_path)

    # Generate submission using the models trained in the previous step
    # This uses the 'test' dataset defined in metadata
    generate_submission(load_cached_data=True)

    assert os.path.exists(submission_path), "Submission file was not generated"

    # Verify submission content format
    df = pd.read_csv(submission_path)
    print(f"Submission shape: {df.shape}")

    assert (
        "id" in df.columns and "value" in df.columns
    ), "Submission missing required columns"
    assert len(df) > 0, "Submission file is empty"

    # Check ID format (e.g., "110_1_1")
    sample_id = df.iloc[0]["id"]
    parts = sample_id.split("_")
    assert (
        len(parts) == 3
    ), f"ID format incorrect. Expected imgId_row_col, got {sample_id}"

    # Check value range
    # Values should be pixel intensities [0, 1] (or 0 and 1 if thresholded, but task description says intensity)
    # The provided code outputs floats clamped to [0, 1].
    assert (
        df["value"].min() >= 0.0 and df["value"].max() <= 1.0
    ), "Submission values out of range [0, 1]"

    print("Inference and submission verified.\n")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(SEED)

    try:
        # 1. Verify utility functions
        test_utils()

        # 2. Verify Data Loading
        test_dataset_loading()

        # 3. Verify Model
        test_model_architecture()

        # 4. Run Training (Integration Test)
        # We limit to 2 folds and 1 epoch for speed
        test_training_process()

        # 5. Run Inference (Integration Test)
        test_inference_and_submission()

        print("All tests passed successfully!")

    except Exception as e:
        print(f"\nFAILED: An error occurred during execution: {e}")
        raise e
