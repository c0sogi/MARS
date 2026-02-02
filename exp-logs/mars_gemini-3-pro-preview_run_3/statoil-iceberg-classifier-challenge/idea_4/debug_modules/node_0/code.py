import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import DEVICE, BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, WORKING_DIR
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import IcebergSECNN
from library.train import train_one_epoch, validate
from library.predict import predict_with_tta


def demonstrate_data_loading():
    print("\n=== Demonstrating Data Loading ===")

    # Use debug mode to load a small subset of data for speed
    debug_size = 32
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=True,
        debug=True,
        debug_size=debug_size,
    )

    print(f"Train Loader Length: {len(train_loader)}")
    print(f"Val Loader Length: {len(val_loader)}")
    print(f"Test Loader Length: {len(test_loader)}")

    # Fetch a single batch from the training loader
    images, angles, labels = next(iter(train_loader))

    # Validate Shapes
    # Expected: (Batch, 3, 75, 75)
    assert images.shape == (
        8,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ), f"Incorrect image batch shape: {images.shape}"

    # Expected: (Batch,)
    assert angles.shape == (8,), f"Incorrect angle batch shape: {angles.shape}"

    # Expected: (Batch,)
    assert labels.shape == (8,), f"Incorrect label batch shape: {labels.shape}"

    # Validate Data Types
    assert images.dtype == torch.float32, "Images should be float32"
    assert angles.dtype == torch.float32, "Angles should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print("Data loading and batch shapes verified successfully.")
    return train_loader, val_loader, test_loader


def demonstrate_model_architecture():
    print("\n=== Demonstrating Model Architecture ===")

    model = IcebergSECNN().to(DEVICE)

    # Create dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(
        DEVICE
    )
    dummy_angles = torch.randn(batch_size).to(DEVICE)

    # Forward pass
    output = model(dummy_images, dummy_angles)

    # Validate Output
    # Expected: (Batch, 1) - Binary classification probability
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"

    # Validate Value Range (Sigmoid output should be between 0 and 1)
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output values out of range [0, 1]"

    print("Model architecture and forward pass verified successfully.")
    return model


def demonstrate_training_loop(model, train_loader, val_loader):
    print("\n=== Demonstrating Training Loop ===")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # Run for 1 epoch
    print("Running training for 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    print(f"Training Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Run validation
    print("Running validation...")
    val_loss, val_metric = validate(model, val_loader, criterion, DEVICE)

    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation Metric (LogLoss): {val_metric:.6f}")

    assert not np.isnan(val_loss), "Validation loss returned NaN"
    assert not np.isnan(val_metric), "Validation metric returned NaN"

    print("Training and validation loops verified successfully.")


def demonstrate_inference(model, test_loader):
    print("\n=== Demonstrating Inference with TTA ===")

    # Predict using Test-Time Augmentation
    predictions = predict_with_tta(model, test_loader, DEVICE)

    # Validate Predictions
    # The test loader in debug mode (size 32) should yield 32 predictions
    # Note: get_dataloaders with debug=True slices the test set to debug_size
    expected_count = 32

    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert (
        len(predictions) == expected_count
    ), f"Expected {expected_count} predictions, got {len(predictions)}"
    assert predictions.ndim == 1 or (
        predictions.ndim == 2 and predictions.shape[1] == 1
    ), f"Unexpected prediction shape: {predictions.shape}"

    # Flatten if necessary for check
    flat_preds = predictions.flatten()
    assert (flat_preds >= 0).all() and (
        flat_preds <= 1
    ).all(), "Predictions contain values outside [0, 1]"

    print(f"Generated {len(predictions)} predictions.")
    print("Inference verified successfully.")


if __name__ == "__main__":
    # 1. Set Seed for Reproducibility
    set_seed(42)

    # 2. Data Loading
    train_loader, val_loader, test_loader = demonstrate_data_loading()

    # 3. Model Instantiation
    model = demonstrate_model_architecture()

    # 4. Training Loop
    demonstrate_training_loop(model, train_loader, val_loader)

    # 5. Inference
    demonstrate_inference(model, test_loader)

    print("\nAll demonstrations completed successfully.")
