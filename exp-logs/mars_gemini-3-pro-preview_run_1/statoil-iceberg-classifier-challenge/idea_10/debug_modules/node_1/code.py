import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Ensure the library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library import utils, data, model, engine


def run_demo():
    print("=== Starting Demonstration of Iceberg Classification Pipeline ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up configuration and seeding...")
    utils.seed_everything(Config.SEED)

    # Override Config for faster execution during this demo
    Config.N_FOLDS = 2
    Config.CALIBRATION_EPOCHS = 1
    Config.BATCH_SIZE = 16  # Smaller batch size for quick iteration

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading and Processing
    print("\n[2] Testing Data Loading and Processing...")

    # This will process raw JSONs -> numpy arrays and cache them
    train_data_dict, test_data_dict = data.load_data(load_cached_data=True)

    # Assertions for data integrity
    assert "images" in train_data_dict
    assert "angles" in train_data_dict
    assert "labels" in train_data_dict
    assert len(train_data_dict["images"]) == len(train_data_dict["labels"])
    assert train_data_dict["images"].shape[1:] == (75, 75, 3)  # Processed bands shape
    print("Data loaded and shapes verified.")

    # 3. DataLoader Verification
    print("\n[3] Testing DataLoaders...")
    train_loader, val_loader = data.get_dataloaders(fold_index=0, full_fit=False)

    # Fetch one batch to verify transforms and shapes
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Expected: (Batch, 3, 224, 224) due to resizing in transforms
    assert images.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)
    print("DataLoader batch shapes verified.")

    # 4. Model Instantiation and Forward Pass
    print("\n[4] Testing Model Architecture...")
    net = model.IcebergResNet18().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    logits = net(images, angles)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1)
    print("Model forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n[5] Testing Training and Validation Engine...")
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.LEARNING_RATE)

    # Run training for 1 epoch
    print("Running train_one_epoch...")
    train_loss = engine.train_one_epoch(
        model=net,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        scheduler=None,
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float)
    assert train_loss > 0

    # Run validation
    print("Running validate...")
    val_loss, val_metric = engine.validate(net, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val LogLoss: {val_metric:.4f}")
    assert isinstance(val_loss, float)
    assert isinstance(val_metric, float)

    # 6. Inference Demonstration
    print("\n[6] Testing Inference (TTA)...")
    test_loader, test_ids = data.get_test_loader()

    # Run prediction on a small subset to save time?
    # The test set is small enough (321 images) to run fully.
    preds = engine.predict_tta(net, test_loader, device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Number of Test IDs: {len(test_ids)}")

    assert len(preds) == len(test_ids)
    assert np.all((preds >= 0) & (preds <= 1))
    print("Inference successful.")

    # 7. Full-Fit Loader Demonstration
    print("\n[7] Testing Full-Fit DataLoader...")
    full_train_loader, full_val_loader = data.get_dataloaders(full_fit=True)

    assert full_train_loader is not None
    assert full_val_loader is None

    # Check size of full loader dataset (should be larger than fold split)
    # Fold split (4 folds) means fold train is 80%, full fit is 100%
    # Note: We set N_FOLDS=2 earlier for this demo, so fold train is 50%.
    fold_len = len(train_loader.dataset)
    full_len = len(full_train_loader.dataset)
    print(f"Fold Train Size: {fold_len}, Full Train Size: {full_len}")
    assert full_len > fold_len
    print("Full-fit loader verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
