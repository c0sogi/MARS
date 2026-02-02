import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library import utils, dataset, model as model_lib
from library import train as train_lib


def demo_pipeline():
    print("--- Starting Demonstration of Iceberg Classifier Pipeline ---")

    # 1. Setup Configuration for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 64  # Small subset
    Config.NUM_EPOCHS = 1
    Config.NUM_FOLDS = 2  # We won't run full CV, but setting valid low number
    Config.BATCH_SIZE = 16
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seeds for reproducibility
    utils.set_seed(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Size: {Config.DEBUG_SIZE}")

    # 2. Data Preparation
    print("\n[2] Loading and Preparing Data...")
    # Force reload to demonstrate processing logic, though caching is handled inside
    data = dataset.prepare_data(load_cached_data=True)

    # Verify Data Structure
    assert "train" in data and "val" in data and "test" in data
    assert "X" in data["train"] and "y" in data["train"] and "angle" in data["train"]

    X_full = data["train"]["X"]
    y_full = data["train"]["y"]
    angle_full = data["train"]["angle"]

    print(f"    Original Train X Shape: {X_full.shape}")
    print(f"    Original Train y Shape: {y_full.shape}")

    # Create a small subset for demonstration (simulating the Debug mode logic)
    X_demo = X_full[: Config.DEBUG_SIZE]
    y_demo = y_full[: Config.DEBUG_SIZE]
    angle_demo = angle_full[: Config.DEBUG_SIZE]

    assert len(X_demo) == Config.DEBUG_SIZE
    print("    Subset created successfully.")

    # 3. Dataset and DataLoader
    print("\n[3] Instantiating Dataset and DataLoader...")
    train_ds = dataset.IcebergDataset(
        X_demo, angle_demo, labels=y_demo, transform=dataset.get_transforms("train")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple main-thread execution in demo
    )

    # Verify Batch
    inputs, angles, targets = next(iter(train_loader))
    print(f"    Batch Input Shape: {inputs.shape}")  # Should be (B, 3, 75, 75)
    print(f"    Batch Angle Shape: {angles.shape}")  # Should be (B,)
    print(f"    Batch Target Shape: {targets.shape}")  # Should be (B,)

    assert inputs.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert targets.shape == (Config.BATCH_SIZE,)
    print("    DataLoader functional and shapes verified.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = model_lib.DualPolarityDropBlockSECNN().to(Config.DEVICE)

    # Verify Model Structure (Basic check)
    print(f"    Model Class: {model.__class__.__name__}")

    # Forward Pass Verification
    print("    Performing dummy forward pass...")
    inputs = inputs.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs, angles)

    print(f"    Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Output shape mismatch!"
    print("    Forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Use the library function to train one epoch
    avg_loss = train_lib.train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        epoch=0,
        total_epochs=1,
    )

    print(f"    Training Epoch Completed. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Loss is NaN!"

    # Validate
    print("    Running Validation...")
    # Using the same loader for validation just to demonstrate the function
    val_loss, val_acc = train_lib.validate(
        model=model, loader=train_loader, criterion=criterion, device=Config.DEVICE
    )
    print(f"    Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # 6. Checkpointing Demonstration
    print("\n[6] Testing Checkpointing...")

    # Save
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": val_loss,
        "fold": 0,
    }
    utils.save_checkpoint(state, is_best=True, fold=0)
    print("    Checkpoint saved.")

    # Load
    # Initialize a new model to verify loading
    new_model = model_lib.DualPolarityDropBlockSECNN().to(Config.DEVICE)
    loaded_checkpoint = utils.load_checkpoint(new_model, None, fold=0, load_best=True)

    assert loaded_checkpoint is not None, "Failed to load checkpoint"
    assert loaded_checkpoint["best_loss"] == val_loss, "Loaded metadata mismatch"

    # Verify weights match
    original_params = dict(model.named_parameters())
    loaded_params = dict(new_model.named_parameters())

    for name, param in original_params.items():
        assert torch.equal(
            param, loaded_params[name]
        ), f"Parameter {name} mismatch after loading!"

    print("    Checkpoint loaded and verified successfully.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    demo_pipeline()
