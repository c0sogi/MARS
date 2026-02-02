import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config, DEVICE
from library.model import ScaleDecoupledDenseNet
from library.loss import MaskedMCRMSELoss
from library.utils import set_seed, GlobalMCRMSE
from library.data import get_dataloaders


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run on (cpu or cuda).
        max_grad_norm: Value for gradient clipping.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, partner_indices, targets) in enumerate(loader):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, partner_indices)

        # Calculate loss (MaskedMCRMSELoss handles column selection internally)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the Global MCRMSE metric.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        device: Device to run on.

    Returns:
        float: The computed MCRMSE score.
    """
    model.eval()
    metric_calculator = GlobalMCRMSE()

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            preds = model(inputs, partner_indices)

            # Update global metric accumulator
            metric_calculator.update(preds, targets)

    return metric_calculator.compute()


def run_training(max_epochs=None, max_samples=None):
    """
    Main function to run the training pipeline.

    Args:
        max_epochs (int, optional): Override Config.NUM_EPOCHS.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        float: Best validation MCRMSE score.
    """
    # Reproducibility
    set_seed(Config.SEED)

    # Configuration overrides
    epochs = max_epochs if max_epochs is not None else Config.NUM_EPOCHS

    # Data Loading
    print("Initializing Data Loaders...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, max_samples=max_samples
    )

    # Model Setup
    print("Initializing Model...")
    model = ScaleDecoupledDenseNet().to(DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # Loss Function
    criterion = MaskedMCRMSELoss()

    # Training Loop State
    best_val_score = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_score = validate(model, val_loader, DEVICE)

        # Scheduler Step
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val MCRMSE: {val_score}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training finished. Best Validation MCRMSE: {best_val_score}")
    return best_val_score
