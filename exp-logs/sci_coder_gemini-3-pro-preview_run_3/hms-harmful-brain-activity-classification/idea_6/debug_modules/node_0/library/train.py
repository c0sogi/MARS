import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import seed_everything, KLDivLossWithLogits
from library.data_loader import get_dataloaders
from library.models import DualStreamNetwork


def get_optimizer(model):
    """
    Configures the optimizer with differential learning rates.
    Stream A (Scratch 1D-ResNet) -> Higher LR
    Stream B (Pretrained EfficientNet) -> Lower LR
    Classifier Head -> Higher LR
    """
    # Separate parameters by module
    stream_a_params = list(model.stream_a.parameters())
    stream_b_params = list(model.stream_b.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.AdamW(
        [
            {
                "params": stream_a_params,
                "lr": Config.LR_STREAM_A,
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": stream_b_params,
                "lr": Config.LR_STREAM_B,
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": classifier_params,
                "lr": Config.LR_STREAM_A,  # Head needs to learn quickly
                "weight_decay": Config.WEIGHT_DECAY,
            },
        ]
    )
    return optimizer


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (eeg, spec, targets) in enumerate(loader):
        # Move data to device
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(eeg, spec)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * eeg.size(0)
        count += eeg.size(0)

    # Step scheduler at the end of epoch if it's not None
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / count
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Runs validation loop.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (eeg, spec, targets) in enumerate(loader):
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(eeg, spec)
            loss = criterion(logits, targets)

            running_loss += loss.item() * eeg.size(0)
            count += eeg.size(0)

    val_loss = running_loss / count
    return val_loss


def run_training():
    """
    Main orchestration function for training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(debug=Config.DEBUG)

    # 3. Model Initialization
    print("Initializing DualStreamNetwork...")
    model = DualStreamNetwork(
        num_classes=Config.N_CLASSES, pretrained=Config.PRETRAINED
    )
    model = model.to(device)

    # 4. Optimization Setup
    optimizer = get_optimizer(model)
    criterion = KLDivLossWithLogits()

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation Loss: {best_val_loss}")
