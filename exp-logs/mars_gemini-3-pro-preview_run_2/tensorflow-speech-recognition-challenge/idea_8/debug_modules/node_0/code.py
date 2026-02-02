import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import INPUT_ROOT, METADATA_DIR, NUM_CLASSES, DEVICE, WORKING_DIR
from library.utils import set_seed, calculate_accuracy, AverageMeter
from library.dataset import (
    SpeechCommandDataset,
    _compute_waveforms,
    _compute_labels,
    _compute_fnames,
)
from library.model import MultiScaleEfficientNet
from library.trainer import Trainer


def run_demo():
    print("=== Starting Library Demo ===")

    # 1. Setup and Configuration
    # Ensure reproducibility
    set_seed(42)
    demo_working_dir = os.path.join(WORKING_DIR, "demo_run")
    os.makedirs(demo_working_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Input Root: {INPUT_ROOT}")

    # 2. Data Loading (Optimized for Speed)
    # Instead of loading the full dataset (which takes time to process waveforms),
    # we load a small subset of the metadata and process it on the fly.

    print("\n--- Preparing Data Subset ---")
    train_metadata_path = os.path.join(METADATA_DIR, "train.csv")

    if not os.path.exists(train_metadata_path):
        raise FileNotFoundError(f"Metadata not found at {train_metadata_path}")

    df_full = pd.read_csv(train_metadata_path)
    # Sample 64 items for a quick demo (2 batches of 32)
    df_subset = df_full.sample(n=64, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df_subset)} files from training metadata.")

    # Compute data arrays using library internal functions
    print("Computing waveforms and labels for subset...")
    waveforms = _compute_waveforms(df_subset, INPUT_ROOT)
    labels = _compute_labels(df_subset)
    fnames = _compute_fnames(df_subset)

    # Verify shapes
    print(f"Waveforms shape: {waveforms.shape}")
    print(f"Labels shape: {labels.shape}")

    assert waveforms.ndim == 2 and waveforms.shape[0] == 64, "Waveform shape mismatch"
    assert labels.ndim == 1 and labels.shape[0] == 64, "Label shape mismatch"

    # Create Datasets
    # We use the same subset for train and val to verify overfitting/mechanics quickly
    train_dataset = SpeechCommandDataset(waveforms, labels, fnames, is_train=True)
    val_dataset = SpeechCommandDataset(waveforms, labels, fnames, is_train=False)

    # Create DataLoaders
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Check one batch
    inputs, targets, names = next(iter(train_loader))
    print(f"Batch Input Shape: {inputs.shape}")  # Expected: (B, 1, n_mels, time)
    print(f"Batch Target Shape: {targets.shape}")

    assert inputs.dim() == 4, "Input tensor must be 4D (B, C, F, T)"
    assert inputs.shape[1] == 1, "Input channel must be 1"

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = MultiScaleEfficientNet(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Verify forward pass
    with torch.no_grad():
        dummy_out = model(inputs.to(DEVICE))
    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (batch_size, NUM_CLASSES), "Model output shape mismatch"

    # 4. Training Setup
    print("\n--- Setting up Trainer ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=DEVICE,
        working_dir=demo_working_dir,
    )

    # 5. Execution
    print("\n--- Running Training Epoch ---")
    train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch_idx=1)
    print(f"Train Result -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    print("\n--- Running Validation ---")
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"Val Result   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Basic logic checks
    assert train_loss > 0, "Training loss should be positive"
    assert 0 <= train_acc <= 1, "Training accuracy should be between 0 and 1"

    # 6. Utility Verification
    print("\n--- Verifying Utilities ---")
    # Test calculate_accuracy manually
    logits = torch.tensor([[2.0, 0.5, 0.5], [0.5, 2.0, 0.5]])  # Class 0, Class 1
    true_labels = torch.tensor([0, 1])
    acc = calculate_accuracy(logits, true_labels)
    print(f"Manual Accuracy Test: {acc}")
    assert acc == 1.0, "Accuracy calculation logic failed"

    # Test AverageMeter
    meter = AverageMeter("Test")
    meter.update(10, n=2)
    meter.update(20, n=2)
    print(f"AverageMeter Test: {meter.avg}")
    assert meter.avg == 15.0, "AverageMeter logic failed"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
