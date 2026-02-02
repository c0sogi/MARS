import os
import torch
import numpy as np
import pandas as pd
import sys

# Import components from the provided library
from library.utils import set_seed, SUBMISSION_DIR
from library.dataset import get_dataloaders
from library.coordinate_attention import CoordinateAttention
from library.model import WideCoordinateResNeXt
from library.engine import run_training_and_inference


def demo_data_loading():
    print("\n=== Demonstrating Data Loading ===")
    batch_size = 8
    # Get loaders with 0 workers to avoid multiprocessing overhead in this quick demo
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=0
    )

    # 1. Inspect Train Batch
    print("Fetching batch from Train Loader...")
    images, labels = next(iter(train_loader))

    # Verify Shapes
    # Expected: (Batch, 3, 32, 32)
    assert images.shape == (
        batch_size,
        3,
        32,
        32,
    ), f"Incorrect image shape: {images.shape}"
    # Expected: (Batch,)
    assert labels.shape == (batch_size,), f"Incorrect label shape: {labels.shape}"

    # Verify Data Range (Normalization)
    print(f"Image stats - Min: {images.min():.4f}, Max: {images.max():.4f}")
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images not normalized to [0, 1]"

    # 2. Inspect Test Batch
    print("Fetching batch from Test Loader...")
    test_images, test_ids = next(iter(test_loader))

    # Test loader in library uses batch_size=1 by default
    # Expected: (1, 3, 32, 32)
    assert test_images.shape == (
        1,
        3,
        32,
        32,
    ), f"Incorrect test image shape: {test_images.shape}"
    # Verify ID type
    assert isinstance(test_ids[0], str), "Test IDs should be strings"

    print("Data loading verification successful.")
    return images  # Return a batch for model testing


def demo_model_components(sample_batch):
    print("\n=== Demonstrating Model Components ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = sample_batch.to(device)

    # 1. Coordinate Attention Block
    print("Testing CoordinateAttention Block...")
    # Input channels=3 for raw image, reduction=1 just for shape test
    ca_block = CoordinateAttention(inp=3, reduction=1).to(device)
    out = ca_block(sample_batch)

    assert (
        out.shape == sample_batch.shape
    ), f"CoordinateAttention changed shape: {out.shape}"
    print("CoordinateAttention forward pass successful.")

    # 2. Full WideCoordinateResNeXt Model
    print("Testing WideCoordinateResNeXt Model...")
    model = WideCoordinateResNeXt(cardinality=4).to(
        device
    )  # Reduced cardinality for demo speed

    # Forward pass
    logits = model(sample_batch)

    # Expected output: (Batch, 1) for binary classification logits
    assert logits.shape == (
        sample_batch.size(0),
        1,
    ), f"Model output shape mismatch: {logits.shape}"

    print("Model forward pass successful.")


def demo_full_pipeline():
    print("\n=== Demonstrating Full Training & Inference Pipeline ===")

    # Run the engine with minimal settings for speed
    # 1 Epoch, 1 Seed, larger batch size for faster epoch completion
    run_training_and_inference(epochs=1, batch_size=128, seeds=[42], patience=1)

    # Verify Submission Output
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    df = pd.read_csv(submission_path)
    print(f"\nSubmission file loaded. Shape: {df.shape}")

    # Verify headers
    expected_cols = ["id", "has_cactus"]
    assert list(df.columns) == expected_cols, f"Invalid columns: {df.columns}"

    # Verify values
    assert (
        df["has_cactus"].min() >= 0.0 and df["has_cactus"].max() <= 1.0
    ), "Predictions out of probability range"

    print("Pipeline execution and submission generation successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Validate Data Loading
    # We retrieve a sample batch to use in the model test
    sample_images = demo_data_loading()

    # 2. Validate Model Logic
    demo_model_components(sample_images)

    # 3. Run End-to-End Pipeline
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
