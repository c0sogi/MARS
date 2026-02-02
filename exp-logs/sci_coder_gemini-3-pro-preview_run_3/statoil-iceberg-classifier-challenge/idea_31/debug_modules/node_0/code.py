import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import set_seed, get_device, IcebergDataset, BHA_ResNet, predict_test
from library.dataset import load_data
from library.train import fit_fold


def demo_pipeline():
    # 1. Setup
    print("--- 1. Setup ---")
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Define working directory for this demo
    demo_dir = "./working/demo_usage"
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Data Loading (Subset for Speed)
    print("\n--- 2. Data Loading ---")
    # We load only 100 samples to keep the demo fast
    X_train, y_train, angle_train, X_test, ids_test, angle_test = load_data(
        base_dir=os.path.join(demo_dir, "cache"),
        load_cached_data=False,  # Force processing to demonstrate pipeline
        dataset_size=100,
    )

    # Assertions to verify data loading
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # X should be (N, 3, 75, 75)
    assert X_train.ndim == 4
    assert X_train.shape[1] == 3
    assert X_train.shape[2] == 75 and X_train.shape[3] == 75
    assert len(X_train) == 100

    # y should be (N,)
    assert y_train.ndim == 1
    assert len(y_train) == 100

    # 3. Dataset & DataLoader
    print("\n--- 3. Dataset & DataLoader ---")
    # Split 80/20 for demo
    split_idx = int(0.8 * len(X_train))

    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    angle_tr, angle_val = angle_train[:split_idx], angle_train[split_idx:]

    # Create Datasets
    train_ds = IcebergDataset(X_tr, angle_tr, y_tr, transform=None)
    val_ds = IcebergDataset(X_val, angle_val, y_val, transform=None)

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Verify a single batch
    images, angles, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Should be [8, 3, 75, 75]
    print(f"Batch angles shape: {angles.shape}")  # Should be [8]
    print(f"Batch labels shape: {labels.shape}")  # Should be [8]

    assert images.shape == (batch_size, 3, 75, 75)
    assert angles.shape == (batch_size,)
    assert labels.shape == (batch_size,)

    # 4. Model Instantiation & Forward Pass
    print("\n--- 4. Model Check ---")
    model = BHA_ResNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = model(images, angles)
    print(f"Model output shape: {outputs.shape}")

    # Output should be [Batch_Size, 1] (logits)
    assert outputs.shape == (batch_size, 1)

    # 5. Training Demonstration
    print("\n--- 5. Training Loop (2 Epochs) ---")
    checkpoint_dir = os.path.join(demo_dir, "checkpoints")

    # Train for 2 epochs using the library function
    best_model_path, best_loss = fit_fold(
        fold_idx=0,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        patience=1,
        lr=1e-3,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"Training complete. Best loss: {best_loss}")
    print(f"Checkpoint saved at: {best_model_path}")

    assert os.path.exists(best_model_path), "Checkpoint file was not created."

    # 6. Inference Demonstration
    print("\n--- 6. Inference ---")
    # Load the best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare test loader
    test_ds = IcebergDataset(X_test, angle_test, y=None, transform=None)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Run inference using library utility
    # predict_test expects a list of models (ensemble), so we wrap our single model in a list
    preds = predict_test([model], test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    print(f"First 5 predictions: {preds[:5]}")

    # Assertions on predictions
    assert len(preds) == len(X_test)
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities between 0 and 1"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    demo_pipeline()
