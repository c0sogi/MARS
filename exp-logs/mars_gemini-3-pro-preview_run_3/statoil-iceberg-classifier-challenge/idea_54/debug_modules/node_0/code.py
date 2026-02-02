import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import provided library components
from library.utils import set_seed, get_device
from library.dataset import IcebergDataset
from library.model import SAICNN
from library.train import train_one_epoch, evaluate, predict


def demo_dataset_usage():
    print("--- 1. Demonstrating IcebergDataset ---")
    metadata_path = "./metadata/train.csv"

    # Instantiate dataset in training mode
    # load_cached_data=True will use existing .npy files in ./working/idea_54 if available,
    # otherwise it processes the JSONs from ./input
    dataset = IcebergDataset(metadata_path, mode="train", load_cached_data=True)

    print(f"Dataset size: {len(dataset)}")

    # Fetch a single sample
    # Returns: ((image, angle), label)
    (image, angle), label = dataset[0]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Angle: {angle.item():.4f}")
    print(f"Sample Label: {label.item()}")

    # Validation
    assert len(dataset) > 0, "Dataset should not be empty."
    assert image.shape == (
        3,
        75,
        75,
    ), f"Expected image shape (3, 75, 75), got {image.shape}"
    assert isinstance(angle, torch.Tensor), "Angle should be a tensor."
    assert isinstance(label, torch.Tensor), "Label should be a tensor."

    return dataset


def demo_model_usage():
    print("\n--- 2. Demonstrating SAICNN Model ---")
    device = get_device()
    model = SAICNN().to(device)

    # Create dummy input batch
    batch_size = 4
    # Image: (B, 3, 75, 75)
    dummy_images = torch.randn(batch_size, 3, 75, 75).to(device)
    # Angle: (B,)
    dummy_angles = torch.randn(batch_size).to(device)

    # Forward pass
    output = model(dummy_images, dummy_angles)

    print(f"Input batch size: {batch_size}")
    print(f"Output shape: {output.shape}")

    # Validation
    # Output should be logits of shape (B, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape ({batch_size}, 1), got {output.shape}"

    return model


def demo_training_functions(dataset, model):
    print("\n--- 3. Demonstrating Training Functions ---")
    device = get_device()

    # Create a small DataLoader
    # We use a small batch size to ensure multiple steps even with a subset if needed
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Setup Criterion and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print("Running training for 1 epoch...")
    train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.6f}")

    print("Running evaluation...")
    val_loss = evaluate(model, loader, criterion, device)
    print(f"Evaluation Loss: {val_loss:.6f}")

    # Validation
    assert not np.isnan(train_loss), "Training loss returned NaN."
    assert not np.isnan(val_loss), "Evaluation loss returned NaN."


def demo_inference_usage():
    print("\n--- 4. Demonstrating Inference ---")
    device = get_device()
    metadata_path = "./metadata/test.csv"

    # Instantiate dataset in test mode (returns ID instead of label)
    test_dataset = IcebergDataset(metadata_path, mode="test", load_cached_data=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize a fresh model
    model = SAICNN().to(device)
    model.eval()

    print("Generating predictions...")
    ids, preds = predict(model, test_loader, device)

    print(f"Number of predictions: {len(preds)}")
    print(f"Sample IDs: {ids[:3]}")
    print(f"Sample Probabilities: {preds[:3]}")

    # Validation
    assert len(ids) == len(
        test_dataset
    ), "Mismatch between dataset size and prediction count."
    assert len(preds) == len(
        test_dataset
    ), "Mismatch between dataset size and prediction count."
    # Predictions should be probabilities (0 to 1) because predict() applies sigmoid
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]."


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Run demonstrations
    ds = demo_dataset_usage()
    model = demo_model_usage()
    demo_training_functions(ds, model)
    demo_inference_usage()

    print("\nAll demonstrations completed successfully.")
