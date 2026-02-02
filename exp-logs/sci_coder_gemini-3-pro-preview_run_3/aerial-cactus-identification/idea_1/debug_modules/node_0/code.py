import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.dataset import load_and_preprocess_data, CactusDataset, get_transforms
from library.model import SimpleCNN
from library.train import train_cactus_classifier


def verify_data_components():
    """
    Demonstrates and verifies data loading and dataset creation.
    """
    print("--- Verifying Data Loading and Dataset Components ---")

    # Define paths
    metadata_path = "./metadata/val_metadata.csv"
    input_dir = "./input"
    cache_name = "val_demo"  # Unique name to avoid conflict with main training cache

    # 1. Test raw data loading
    # We set load_cached_data=False to force reading from disk for this demonstration
    ids, images, labels = load_and_preprocess_data(
        metadata_path, input_dir, cache_name, load_cached_data=False
    )

    print(f"Loaded {len(ids)} samples.")

    # Assertions for raw data
    assert len(ids) == len(images) == len(labels), "Mismatch in data arrays length."
    assert images.shape[1:] == (32, 32, 3), f"Unexpected image shape: {images.shape}"
    assert images.dtype == np.uint8, "Images should be loaded as uint8."

    # 2. Test Dataset class
    transform = get_transforms("val")
    dataset = CactusDataset(images, labels, transform=transform)

    # Retrieve a single sample
    img_tensor, label_tensor = dataset[0]

    # Assertions for Dataset output
    assert isinstance(
        img_tensor, torch.Tensor
    ), "Dataset should return a Tensor for image."
    assert img_tensor.shape == (
        3,
        32,
        32,
    ), f"Tensor shape should be (3, 32, 32), got {img_tensor.shape}"
    assert img_tensor.dtype == torch.float32, "Tensor should be float32."
    assert (
        0.0 <= img_tensor.min() and img_tensor.max() <= 1.0
    ), "Image tensor not normalized to [0, 1]."
    assert isinstance(label_tensor, torch.Tensor), "Label should be a Tensor."

    print("Data components verified successfully.\n")


def verify_model_architecture():
    """
    Demonstrates and verifies the model architecture.
    """
    print("--- Verifying Model Architecture ---")

    device = get_device()
    model = SimpleCNN().to(device)

    # Create a dummy batch: (Batch Size, Channels, Height, Width)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Assertions
    assert output.shape == (batch_size, 1), "Output shape mismatch."
    # Check if sigmoid is applied (values between 0 and 1)
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Output values outside [0, 1] range."

    print("Model architecture verified successfully.\n")


def run_training_pipeline():
    """
    Demonstrates the full training pipeline using the library function.
    """
    print("--- Running Full Training Pipeline ---")

    submission_dir = "./submission"

    # Ensure clean state for submission
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)

    # Execute training
    # We use minimal epochs and patience for a quick demonstration
    train_cactus_classifier(
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        patience=1,
        load_cached_data=True,
        input_dir="./input",
        metadata_dir="./metadata",
        submission_dir=submission_dir,
    )

    # Verify Submission File
    submission_file = os.path.join(submission_dir, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not created."

    df = pd.read_csv(submission_file)
    print(f"Submission file created with shape: {df.shape}")

    # Check format
    assert list(df.columns) == [
        "id",
        "has_cactus",
    ], "Incorrect columns in submission file."
    assert df["has_cactus"].dtype == float, "Prediction column should be float."
    assert len(df) > 0, "Submission file is empty."

    print("Training pipeline completed and verified successfully.\n")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    # Run demonstrations
    verify_data_components()
    verify_model_architecture()
    run_training_pipeline()

    print("All demonstrations passed.")
