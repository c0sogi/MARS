import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config, seed_everything
from library.dataset import DogCatDataset, get_transforms
from library.models import create_model
from library.utils import AverageMeter, accuracy
from library.engine import train_model
from library.inference import ensemble_predictions


def create_mini_metadata():
    """
    Creates small subsets of the original metadata files to allow for
    rapid execution of the training and inference pipelines.
    """
    print("Creating mini metadata for demonstration...")

    # Ensure working directory exists
    demo_dir = "./working/demo_metadata"
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Sample subsets (20 train, 10 val, 10 test)
    mini_train = train_df.head(20).copy()
    mini_val = val_df.head(10).copy()
    mini_test = test_df.head(10).copy()

    # Save to demo directory
    train_path = os.path.join(demo_dir, "mini_train.csv")
    val_path = os.path.join(demo_dir, "mini_val.csv")
    test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train.to_csv(train_path, index=False)
    mini_val.to_csv(val_path, index=False)
    mini_test.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def test_utils():
    """
    Validates utility functions: AverageMeter and accuracy.
    """
    print("Testing Utility functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    assert meter.count == 2, f"AverageMeter count failed: expected 2, got {meter.count}"

    # Test Accuracy
    # Case 1: All correct
    logits = torch.tensor([10.0, -10.0, 10.0])  # Sigmoid -> ~1, ~0, ~1
    targets = torch.tensor([1.0, 0.0, 1.0])
    acc = accuracy(logits, targets)
    assert acc == 100.0, f"Accuracy failed: expected 100.0, got {acc}"

    # Case 2: None correct
    logits = torch.tensor([-10.0, 10.0])  # Sigmoid -> ~0, ~1
    targets = torch.tensor([1.0, 0.0])
    acc = accuracy(logits, targets)
    assert acc == 0.0, f"Accuracy failed: expected 0.0, got {acc}"

    print("Utils verified.")


def test_dataset(img_size=224):
    """
    Validates the DogCatDataset class.
    """
    print("Testing DogCatDataset...")

    # Instantiate dataset with limit
    dataset = DogCatDataset(
        split="train",
        img_size=img_size,
        transform=get_transforms(img_size, is_train=True),
        limit=5,
    )

    # Check length
    assert len(dataset) == 5, f"Dataset length mismatch: expected 5, got {len(dataset)}"

    # Check item structure
    img, label = dataset[0]

    # Check Image Tensor
    assert isinstance(img, torch.Tensor), "Dataset item image is not a Tensor"
    assert img.shape == (
        3,
        img_size,
        img_size,
    ), f"Image shape mismatch: expected (3, {img_size}, {img_size}), got {img.shape}"

    # Check Label
    assert isinstance(label, torch.Tensor), "Dataset item label is not a Tensor"
    assert label.ndim == 0, "Label should be a scalar tensor"

    print("Dataset verified.")


def test_model_creation():
    """
    Validates model instantiation and forward pass.
    """
    print("Testing Model Creation...")

    model_key = "resnet50"
    model = create_model(model_key, pretrained=False)  # False for speed
    model.eval()

    # Create dummy input
    batch_size = 2
    img_size = Config.MODEL_SPECS[model_key]["img_size"]
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape (Batch_Size, 1) - since num_classes=1
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch: expected ({batch_size}, 1), got {output.shape}"

    print("Model creation verified.")


def run_pipeline_demo():
    """
    Runs the training and inference pipeline using the mini-dataset.
    """
    print("\n--- Running Training Pipeline Demo ---")

    # 1. Train Model
    # We only train one model for the demo to save time
    model_key = "resnet50"
    best_loss = train_model(model_key)

    print(f"Training completed. Best Validation Loss: {best_loss:.4f}")

    # Verify checkpoint exists
    checkpoint_path = os.path.join(Config.WORKING_DIR, f"{model_key}_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 2. Inference
    print("\n--- Running Inference Pipeline Demo ---")

    # Run ensemble predictions (will use the single trained model)
    ensemble_predictions(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in sub_df.columns and "label" in sub_df.columns
    ), "Submission columns missing"
    assert (
        len(sub_df) == 10
    ), f"Submission length mismatch: expected 10, got {len(sub_df)}"
    assert sub_df["label"].between(0, 1).all(), "Predictions out of range [0, 1]"

    print("Pipeline verified successfully.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Create Mini Metadata
    mini_train_path, mini_val_path, mini_test_path = create_mini_metadata()

    # 3. Patch Config for Demo
    # We modify the Config class attributes directly to influence the library behavior
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Optimize hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4

    # Limit to a single model for the demo
    Config.MODEL_SPECS = {
        "resnet50": {
            "model_name": "resnet50.a1_in1k",
            "img_size": 224,  # Use 224 for speed in demo
            "pretrained": True,
        }
    }

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 4. Run Validations
    test_utils()
    test_dataset(img_size=224)
    test_model_creation()

    # 5. Run Pipeline
    run_pipeline_demo()

    print("\nAll demonstrations completed successfully.")
