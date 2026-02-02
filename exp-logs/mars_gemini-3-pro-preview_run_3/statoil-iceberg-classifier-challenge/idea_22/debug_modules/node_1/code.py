import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Import Config and override for fast demo execution
from library.config import Config

# Override Config for speed and resource constraints
print("Configuring parameters for fast demonstration...")
Config.DEBUG = True
Config.MAX_SAMPLES = 100  # Use only 100 samples to speed up loading/training
Config.BATCH_SIZE = 16
Config.NUM_EPOCHS = 1  # Run only 1 epoch per fold
Config.NUM_FOLDS = 2  # Run only 2 folds instead of 5
Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in demo

# 2. Import Library Modules
from library.utils import set_seed, save_checkpoint, load_checkpoint
import library.dataset as lib_dataset
from library.dataset import get_dataloaders
from library.model import MASHCNN
from library.trainer import train_one_epoch, validate, train

# 3. Fix Incompatibility between dataset.py and trainer.py
# The provided trainer.py uses `batch["image"]` (dict access) AND `batch[1]` (index/key access).
# The provided dataset.py returns a tuple `(sample_dict, label_tensor)`.
# To make them compatible without editing files, we monkeypatch IcebergDataset.__getitem__
# to return a single dictionary where the label is stored under the integer key 1.
print("Applying compatibility patch to IcebergDataset...")
original_getitem = lib_dataset.IcebergDataset.__getitem__


def patched_getitem(self, idx):
    result = original_getitem(self, idx)
    # If result is a tuple (sample, label), merge them into the sample dict
    if isinstance(result, tuple):
        sample, label = result
        sample[1] = label  # Store label under key 1 to satisfy batch[1] access
        return sample
    return result


lib_dataset.IcebergDataset.__getitem__ = patched_getitem


def run_demo():
    print("\n--- Starting MASH-CNN Demo ---")

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Ensure output directories exist
    Config.setup_directories()

    # --- Part 1: Data Loading ---
    print("\n[1] Testing Data Loading...")
    # This will use the patched dataset class
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Verify batch structure matches our patch
    batch = next(iter(train_loader))
    assert isinstance(batch, dict), "Batch must be a dictionary after patching"
    assert "image" in batch, "Batch must contain 'image' key"
    assert "angle" in batch, "Batch must contain 'angle' key"
    assert 1 in batch, "Batch must contain key 1 (labels)"

    images = batch["image"]
    angles = batch["angle"]
    labels = batch[1]

    print(
        f"Batch Shapes - Image: {images.shape}, Angle: {angles.shape}, Label: {labels.shape}"
    )

    # Verify dimensions
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.INPUT_CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert labels.shape == (Config.BATCH_SIZE,)

    # --- Part 2: Model Instantiation ---
    print("\n[2] Testing Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = MASHCNN().to(device)

    # Forward pass check
    images = images.to(device)
    angles = angles.to(device)

    # Verify forward pass works without error
    outputs = model(images, angles)

    print(f"Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Incorrect output shape"
    assert torch.isfinite(outputs).all(), "Model produced NaN/Inf values"

    # --- Part 3: Manual Training Step ---
    print("\n[3] Testing Single Training Step...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch manually using the library function
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.6f}")
    assert np.isfinite(train_loss), "Training loss is not finite"

    # Run validation manually
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.6f}")
    assert np.isfinite(val_loss), "Validation loss is not finite"

    # --- Part 4: Checkpointing ---
    print("\n[4] Testing Checkpoint System...")
    # Save current state
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "best_metric": val_loss,
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(state, is_best=True, fold=0)

    # Load into a new model instance
    new_model = MASHCNN().to(device)
    loaded_epoch, loaded_metric = load_checkpoint(
        fold=0, model=new_model, device=device
    )

    print(f"Loaded Checkpoint - Epoch: {loaded_epoch}, Metric: {loaded_metric:.6f}")
    assert loaded_epoch == 1
    assert abs(loaded_metric - val_loss) < 1e-6

    # Verify weights match
    p1 = next(model.parameters())
    p2 = next(new_model.parameters())
    assert torch.equal(p1, p2), "Weights mismatch after loading checkpoint"

    # --- Part 5: Full Trainer Integration ---
    print("\n[5] Testing Full Trainer (Short CV)...")
    # This runs the full cross-validation logic defined in trainer.py
    # using our patched dataset and reduced Config settings.
    # It will run 2 folds, 1 epoch each.
    try:
        train()
        print("Trainer execution completed successfully.")
    except Exception as e:
        print(f"Trainer execution failed: {e}")
        raise e

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
