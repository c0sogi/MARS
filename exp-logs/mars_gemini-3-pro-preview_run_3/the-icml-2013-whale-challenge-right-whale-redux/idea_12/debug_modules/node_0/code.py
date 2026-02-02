import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import process_data, WhaleDataset, get_transforms
from library.transforms import LogMelSpectrogram, InstanceNorm, SpecAugment
from library.model import WhaleEfficientNet
from library.loss import WeightedBCELoss
from library.trainer import Trainer


def run_demo():
    print("=== Starting Right Whale Detection Library Demo ===")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration and Seeds...")

    # Override Config for the demo to ensure speed
    class DemoConfig(Config):
        WORKING_DIR = "./working/demo_execution"
        BATCH_SIZE = 8
        EPOCHS = 2
        BACKBONE = "tf_efficientnet_b0"  # Smaller backbone for speed
        NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
        PATIENCE = 2

    # Create working directory
    if os.path.exists(DemoConfig.WORKING_DIR):
        shutil.rmtree(DemoConfig.WORKING_DIR)
    os.makedirs(DemoConfig.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(DemoConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {DemoConfig.WORKING_DIR}")

    # 2. Data Processing
    # -------------------------------------------------------------------------
    print("\n[2] Processing Data (Subset)...")

    # Process a small subset of training data (32 samples)
    # We use a unique cache name 'train_demo' to avoid overwriting full training cache
    train_specs, train_labels, train_clips = process_data(
        metadata_path=DemoConfig.TRAIN_METADATA,
        cache_name="train_demo",
        load_cached=False,
        max_samples=32,
    )

    # Process a small subset of validation data (16 samples)
    val_specs, val_labels, val_clips = process_data(
        metadata_path=DemoConfig.VAL_METADATA,
        cache_name="val_demo",
        load_cached=False,
        max_samples=16,
    )

    print(f"    Train Specs Shape: {train_specs.shape}")
    print(f"    Val Specs Shape:   {val_specs.shape}")

    # Assertions for data loading
    assert len(train_specs) == 32, "Train subset size mismatch"
    assert len(val_specs) == 16, "Val subset size mismatch"
    assert train_specs.ndim == 3, "Spectrograms should be (N, Freq, Time)"
    # Check if spectrograms are normalized (roughly in 0-1 range due to InstanceNorm)
    assert (
        train_specs.max() <= 1.0 + 1e-5
    ), "Spectrograms not properly normalized (max > 1)"
    assert (
        train_specs.min() >= 0.0 - 1e-5
    ), "Spectrograms not properly normalized (min < 0)"

    # 3. Component Verification: Transforms & Dataset
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Transforms and Dataset...")

    # Test Transforms directly
    dummy_waveform = torch.randn(1, DemoConfig.N_SAMPLES)
    mel_transform = LogMelSpectrogram()
    spec = mel_transform(dummy_waveform)
    print(f"    LogMelSpectrogram Output Shape: {spec.shape}")

    # Test Dataset
    train_transform = get_transforms(mode="train")
    train_dataset = WhaleDataset(train_specs, train_labels, transform=train_transform)

    # Fetch one sample
    sample_spec, sample_target = train_dataset[0]

    print(f"    Dataset Item Shape: {sample_spec.shape}")
    print(f"    Dataset Target: {sample_target}")

    # Assertions
    # Dataset should expand 1-channel spec to 3-channel for EfficientNet
    assert sample_spec.shape[0] == 3, "Dataset did not expand to 3 channels"
    assert sample_spec.shape[1] == DemoConfig.N_MELS, "Incorrect Mel bins"
    assert isinstance(sample_target, torch.Tensor), "Target is not a tensor"

    # 4. Component Verification: Model & Loss
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model and Loss...")

    # Initialize Model
    model = WhaleEfficientNet(backbone_name=DemoConfig.BACKBONE, pretrained=False)
    model.to(device)

    # Test Forward Pass with dummy batch
    dummy_batch = torch.randn(2, 3, DemoConfig.N_MELS, sample_spec.shape[2]).to(device)
    with torch.no_grad():
        logits = model(dummy_batch)

    print(f"    Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (2, 1), f"Expected output shape (2, 1), got {logits.shape}"

    # Test Loss
    criterion = WeightedBCELoss(pos_weight=torch.tensor(2.0))
    dummy_targets = torch.tensor([[1.0], [0.0]]).to(device)
    loss = criterion(logits, dummy_targets)

    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Trainer)...")

    # Setup DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DemoConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=DemoConfig.NUM_WORKERS,
    )

    val_dataset = WhaleDataset(val_specs, val_labels, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=DemoConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=DemoConfig.NUM_WORKERS,
    )

    # Setup Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=DemoConfig.EPOCHS,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=DemoConfig,
    )

    # Run Fit
    save_path = os.path.join(DemoConfig.WORKING_DIR, "checkpoints", "best_model.pth")
    best_auc = trainer.fit(
        train_loader, val_loader, save_path, epochs=DemoConfig.EPOCHS
    )

    print(f"    Training complete. Best Val AUC: {best_auc:.4f}")
    assert os.path.exists(save_path), "Model checkpoint was not saved."

    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference...")

    # Predict on validation set
    probs = trainer.predict(val_loader)

    print(f"    Predictions Shape: {probs.shape}")
    print(f"    First 5 Predictions: {probs[:5]}")

    # Assertions
    assert len(probs) == len(
        val_dataset
    ), "Number of predictions does not match dataset size"
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Calculate final metric manually to verify utility
    final_auc = calculate_roc_auc(val_labels, probs)
    print(f"    Final Verification AUC: {final_auc:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
