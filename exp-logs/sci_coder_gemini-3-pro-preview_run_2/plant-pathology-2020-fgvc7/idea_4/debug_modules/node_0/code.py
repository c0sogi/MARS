import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import AppleDataset, get_transforms
from library.model import AppleClassifier
from library.loss import WeightedLabelSmoothCrossEntropy
from library.train import run_training


def test_dataset_component():
    """
    Verifies that the AppleDataset loads images and labels correctly.
    """
    print("\n[1/5] Testing AppleDataset...")

    # Load a small subset of training metadata
    train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    df = pd.read_csv(train_meta_path).head(10)

    # Initialize dataset
    transforms = get_transforms("train")
    dataset = AppleDataset(df, transforms=transforms, test_mode=False)

    # Fetch one sample
    image, label = dataset[0]

    # Verify Image Tensor
    # Expected shape: (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    expected_shape = (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        image.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {image.shape}"
    assert isinstance(image, torch.Tensor), "Image is not a torch.Tensor"

    # Verify Label Tensor
    # Expected shape: (4,) since there are 4 classes
    assert label.shape == (
        4,
    ), f"Label shape mismatch. Expected (4,), got {label.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a torch.Tensor"

    print("      AppleDataset verification passed.")


def test_model_component():
    """
    Verifies that the AppleClassifier initializes and performs a forward pass.
    """
    print("\n[2/5] Testing AppleClassifier...")

    # Initialize model (pretrained=False for speed in this unit test)
    model = AppleClassifier(pretrained=False)
    model.eval()

    # Create dummy input batch: (Batch_Size, Channels, Height, Width)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    # Verify Output Shape: (Batch_Size, Num_Classes)
    expected_shape = (batch_size, Config.NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {logits.shape}"

    print("      AppleClassifier verification passed.")


def test_loss_component():
    """
    Verifies the WeightedLabelSmoothCrossEntropy loss function.
    """
    print("\n[3/5] Testing Loss Function...")

    # Setup
    batch_size = 4
    num_classes = Config.NUM_CLASSES

    # Dummy logits (predictions) and targets (class indices)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Initialize Loss
    criterion = WeightedLabelSmoothCrossEntropy()

    # Compute Loss
    loss = criterion(logits, targets)

    # Verify Loss is a scalar tensor
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss returned NaN"

    print("      Loss function verification passed.")


def test_metric_component():
    """
    Verifies the ROC AUC calculation logic.
    """
    print("\n[4/5] Testing Metric Calculation...")

    # Case 1: Perfect predictions
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    y_pred = np.array([[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]])

    # Note: ROC AUC with only one positive sample per class in y_true might raise issues
    # or return default values in standard sklearn, but our utils.calculate_metric handles exceptions.
    # However, to get a valid score, we need more samples or mixed classes.
    # Let's construct a case that definitely works mathematically.

    y_true_robust = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred_robust = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    score = calculate_metric(y_true_robust, y_pred_robust)
    assert score == 1.0, f"Expected perfect score 1.0, got {score}"

    print("      Metric calculation verification passed.")


def run_integration_pipeline():
    """
    Runs the full training and inference pipeline using the library's run_training function.
    Uses a debug subset and 1 epoch for speed.
    """
    print("\n[5/5] Running Integration Pipeline (Training & Inference)...")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Run training with debug=True to use a small subset of data
    # and epochs=1 to ensure quick execution.
    # This function handles:
    # 1. Data loading (get_dataloaders)
    # 2. Model initialization
    # 3. Training loop
    # 4. Validation loop
    # 5. Inference on test set
    # 6. Submission file generation

    try:
        run_training(epochs=1, debug=True)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    # Check submission content
    sub_df = pd.read_csv(submission_path)

    # In debug mode, we expect rows equal to Config.DEBUG_SUBSET_SIZE (or less if test set is smaller)
    # The test set has 183 rows. Config.DEBUG_SUBSET_SIZE is 100.
    # So we expect min(183, 100) = 100 rows.
    expected_rows = min(183, Config.DEBUG_SUBSET_SIZE)
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    print(f"      Integration pipeline passed. Submission saved to {submission_path}")


if __name__ == "__main__":
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print("Starting Apple Disease Detection Library Demo...")

    # Run Unit Tests
    test_dataset_component()
    test_model_component()
    test_loss_component()
    test_metric_component()

    # Run Integration Test
    run_integration_pipeline()

    print("\nAll demonstrations and verifications completed successfully.")
