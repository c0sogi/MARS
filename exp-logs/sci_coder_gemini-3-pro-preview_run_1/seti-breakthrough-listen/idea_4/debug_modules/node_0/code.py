import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import set_seed, get_score, AverageMeter
from library.dataset import get_dataloaders
from library.model import LateFusionModel
from library.engine import run_training


def test_utils():
    print("Testing utils.py...")

    # 1. Test set_seed for reproducibility
    set_seed(42)
    rand1 = torch.randn(5)
    set_seed(42)
    rand2 = torch.randn(5)
    assert torch.equal(rand1, rand2), "set_seed failed to ensure reproducibility."

    # 2. Test get_score (ROC AUC)
    y_true = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    score = get_score(y_true, y_pred_perfect)
    assert score == 1.0, f"get_score failed for perfect predictions. Got {score}"

    # Test single class handling (should return 0.5)
    y_true_single = np.array([0, 0, 0, 0])
    score_single = get_score(y_true_single, y_pred_perfect)
    assert score_single == 0.5, f"get_score failed for single class. Got {score_single}"

    # 3. Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)  # sum=20, count=2
    meter.update(20, n=1)  # sum=40, count=3
    assert abs(meter.avg - 13.3333) < 1e-4, f"AverageMeter failed. Got {meter.avg}"

    print("Utils verified successfully.\n")


def test_dataset():
    print("Testing dataset.py...")

    # Use debug mode to load a tiny subset
    batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead in test
        debug=True,
        debug_sample_size=10,
    )

    # Fetch one batch from train_loader
    images, targets = next(iter(train_loader))

    # Verify shapes
    # Expected: (Batch, Time, Channels, Height, Width) -> (4, 6, 1, 273, 256)
    expected_shape = (batch_size, 6, 1, 273, 256)
    assert (
        images.shape == expected_shape
    ), f"Dataset image shape mismatch. Expected {expected_shape}, got {images.shape}"

    # Expected target shape: (Batch,) or (Batch, 1) depending on squeeze/unsqueeze in dataset vs loader
    # The dataset returns a scalar tensor, loader stacks them -> (Batch,)
    assert (
        targets.shape[0] == batch_size
    ), f"Dataset target batch size mismatch. Got {targets.shape}"

    # Verify data types
    assert images.dtype == torch.float32, "Image tensor should be float32"
    assert targets.dtype == torch.float32, "Target tensor should be float32"

    print("Dataset verified successfully.\n")


def test_model():
    print("Testing model.py...")

    # Instantiate model without pretrained weights for speed
    model = LateFusionModel(pretrained=False)
    model.eval()

    # Create dummy input: (Batch, Time, Channels, Height, Width)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 6, 1, 273, 256)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape: (Batch, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"

    # Verify values are finite (no NaNs)
    assert torch.isfinite(output).all(), "Model output contains NaNs or Infs"

    print("Model architecture verified successfully.\n")


def test_engine():
    print("Testing engine.py (Full Training Loop)...")

    # Override Config for speed
    Config.PRETRAINED = False  # Disable downloading ImageNet weights
    Config.WORKING_DIR = "./working/test_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "test_model.pth")
    Config.SUBMISSION_DIR = "./working/test_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up any previous test run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    # Run training with minimal parameters
    run_training(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        patience=1,
        debug=True,  # Uses small subset
        num_workers=0,  # Avoid multiprocessing overhead
    )

    # Verify outputs
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission file missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."

    print("Engine execution verified successfully.\n")


if __name__ == "__main__":
    # Ensure strict reproducibility for the test script
    set_seed(42)

    try:
        test_utils()
        test_dataset()
        test_model()
        test_engine()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)
