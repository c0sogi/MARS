import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the python path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import ModifiedDenseNet
from library.train import run_training
from library.predict import predict_submission


def demo_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n[Demo] Verifying Utilities...")

    # Test ROC AUC calculation
    y_true = np.array([0, 0, 1, 1])
    # Perfect predictions
    y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    auc_perfect = calculate_roc_auc(y_true, y_pred_perfect)
    print(f"  Perfect AUC: {auc_perfect}")
    assert auc_perfect == 1.0, "AUC calculation failed for perfect predictions"

    # Random predictions
    y_pred_random = np.array([0.6, 0.4, 0.6, 0.4])
    auc_random = calculate_roc_auc(y_true, y_pred_random)
    print(f"  Random AUC: {auc_random}")
    assert 0.0 <= auc_random <= 1.0, "AUC must be between 0 and 1"

    print("  PASS: Utilities verified.")


def demo_dataset_and_loaders():
    """
    Demonstrates dataset loading and verifies tensor shapes.
    """
    print("\n[Demo] Verifying Dataset and DataLoaders...")

    # Override Config for speed
    Config.DEBUG_SAMPLE_SIZE = 32
    Config.BATCH_SIZE = 4

    # Initialize DataLoaders in debug mode
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"  Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Check shapes: (Batch, 3, 48, 48) and (Batch,)
    assert images.shape == (
        4,
        3,
        Config.INPUT_SIZE,
        Config.INPUT_SIZE,
    ), "Incorrect train image shape"
    assert labels.shape == (4,), "Incorrect train label shape"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # Verify Test Loader (returns images and IDs)
    test_images, test_ids = next(iter(test_loader))
    print(f"  Test Batch  - Images: {test_images.shape}, IDs: {len(test_ids)}")

    assert test_images.shape == (
        4,
        3,
        Config.INPUT_SIZE,
        Config.INPUT_SIZE,
    ), "Incorrect test image shape"
    assert len(test_ids) == 4, "Incorrect number of test IDs"

    print("  PASS: DataLoaders verified.")


def demo_model_architecture():
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n[Demo] Verifying Model Architecture...")

    # Instantiate model (pretrained=False to avoid download overhead for this check)
    model = ModifiedDenseNet(pretrained=False)
    model.eval()

    # Create dummy input: (Batch=2, Channels=3, H=48, W=48)
    dummy_input = torch.randn(2, 3, Config.INPUT_SIZE, Config.INPUT_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")

    # Check output shape: (Batch, 1) for binary classification logits
    assert output.shape == (2, 1), "Model output shape mismatch"

    print("  PASS: Model architecture verified.")


def demo_training_pipeline():
    """
    Demonstrates the full training loop.
    """
    print("\n[Demo] Running Training Pipeline...")

    # Configure for a very short run
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset
    Config.BATCH_SIZE = 8

    # Ensure working directory is clean/ready
    if os.path.exists(Config.MODEL_PATH):
        os.remove(Config.MODEL_PATH)

    # Run training
    # Note: This will download weights if not cached, but that's expected behavior.
    run_training(debug=True)

    # Verify that the model checkpoint was created
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not created."
    print(f"  Model saved to: {Config.MODEL_PATH}")

    print("  PASS: Training pipeline verified.")


def demo_prediction_pipeline():
    """
    Demonstrates the inference and submission generation pipeline.
    """
    print("\n[Demo] Running Prediction Pipeline...")

    # Ensure submission directory exists
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # Run prediction using the model trained in the previous step
    predict_submission(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Load and validate content
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Shape: {df.shape}")
    print(f"  First 5 rows:\n{df.head()}")

    # Check columns
    assert (
        "id" in df.columns and "label" in df.columns
    ), "Missing required columns in submission."

    # Check value ranges (probabilities)
    assert df["label"].min() >= 0.0, "Probabilities cannot be negative."
    assert df["label"].max() <= 1.0, "Probabilities cannot exceed 1.0."

    # Check that we have predictions for the debug subset
    # Note: TTA might drop last incomplete batch depending on implementation,
    # but here we just check we have rows.
    assert len(df) > 0, "Submission file is empty."

    print("  PASS: Prediction pipeline verified.")


if __name__ == "__main__":
    # 1. Set global seed for reproducibility
    set_seed(42)

    # 2. Verify Utilities
    demo_utils()

    # 3. Verify Data Loading
    demo_dataset_and_loaders()

    # 4. Verify Model Logic
    demo_model_architecture()

    # 5. Verify Training
    demo_training_pipeline()

    # 6. Verify Prediction
    demo_prediction_pipeline()

    print("\nAll demonstrations completed successfully.")
