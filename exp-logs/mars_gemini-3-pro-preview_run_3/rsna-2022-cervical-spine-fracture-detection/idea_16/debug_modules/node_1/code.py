import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

# Import from the provided library
from library.config import Config, seed_everything
from library.data import CervicalDataset, get_transforms
from library.model import FractureMILModel
from library.train import train_one_epoch, validate_one_epoch, FractureLoss
from library.utils import calculate_weighted_loss_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== RSNA Cervical Spine Fracture Detection Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ------------------------------------------------------------------------
    print("[1] Configuring parameters for rapid demonstration...")

    # Override Config defaults to ensure speed and minimal resource usage
    seed_everything(42)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Only use 4 samples
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_SLICES = 8  # Reduce depth from 64 to 8 for speed
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.NUM_WORKERS = 0  # Use main process to avoid multiprocessing overhead in demo
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")

    # Ensure clean cache directory for demo
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Num Slices: {Config.NUM_SLICES}")
    print(f"    Cache Dir : {Config.CACHE_DIR}")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # ------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # Load metadata and sample for debug
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    print(f"    Loaded {len(train_df)} training samples.")

    # Initialize Dataset
    # Note: We disable loading from existing cache to demonstrate processing logic
    train_ds = CervicalDataset(
        train_df, transforms=get_transforms("train"), load_cached_data=False
    )

    # Fetch a single item to verify structure
    print("    Fetching a single sample for verification...")
    images, labels = train_ds[0]

    # Expected Shape: (NUM_SLICES, Channels=3, H=224, W=224)
    expected_shape = (Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Assertions
    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
    assert labels.shape == (
        Config.NUM_CLASSES,
    ), f"Label shape mismatch. Expected ({Config.NUM_CLASSES},), got {labels.shape}"

    print(f"    [Success] Sample Image Shape: {images.shape}")
    print(f"    [Success] Sample Label Shape: {labels.shape}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    val_ds = CervicalDataset(
        val_df, transforms=get_transforms("val"), load_cached_data=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # ------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")

    # Initialize Model
    model = FractureMILModel(config=Config)
    model.to(device)

    # Create dummy input batch on device
    # Shape: (Batch=2, Slices=8, Channels=3, H=224, W=224)
    dummy_input = torch.stack([images, images]).to(device)

    print("    Running forward pass...")
    with torch.no_grad():
        logits = model(dummy_input)

    # Verify Output Shape: (Batch, Num_Classes)
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {logits.shape}"

    print(f"    [Success] Output Logits Shape: {logits.shape}")

    # ------------------------------------------------------------------------
    # 4. Loss Function & Metric Verification
    # ------------------------------------------------------------------------
    print("\n[4] Testing Loss Function & Metric...")

    criterion = FractureLoss()
    dummy_targets = torch.stack([labels, labels]).to(device)

    # Calculate Loss
    loss = criterion(logits, dummy_targets)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss must be positive"

    # Verify Metric Calculation Utility
    print("    Verifying metric logic...")
    # Create synthetic ground truth and predictions
    # Columns: C1..C7, patient_overall
    y_true_synth = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]])
    # Prediction 1: Very accurate
    y_pred_good = np.array(
        [
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99],
        ]
    )
    # Prediction 2: Very inaccurate (inverse)
    y_pred_bad = 1.0 - y_pred_good

    score_good = calculate_weighted_loss_metric(y_true_synth, y_pred_good)
    score_bad = calculate_weighted_loss_metric(y_true_synth, y_pred_bad)

    print(f"    Good Prediction Score: {score_good:.4f}")
    print(f"    Bad Prediction Score : {score_bad:.4f}")

    assert (
        score_good < score_bad
    ), "Metric logic failed: Good prediction should have lower loss."
    print("    [Success] Metric logic verified.")

    # ------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # ------------------------------------------------------------------------
    print("\n[5] Simulating Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train Step
    print("    Training...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=0
    )
    print(f"    -> Train Epoch Loss: {train_loss:.4f}")

    # Validation Step
    print("    Validating...")
    val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, device)
    print(f"    -> Val Epoch Loss: {val_loss:.4f}")
    print(f"    -> Val Metric: {val_metric:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
