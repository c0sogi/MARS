import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.model import BHAResNet
from library.data_loader import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, angles, targets) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape [B, 1]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(model, loader, criterion, device):
    """
    Validation loop. Returns average loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, angles, targets) in enumerate(loader):
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches


def train_model(fold=0, epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE):
    """
    Main function to train the model with early stopping.

    Args:
        fold (int): Fold index for checkpoint naming.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # Initialize Model
    model = BHAResNet()
    model = model.to(device)

    # Optimizer and Loss
    # Using AdamW with constant learning rate as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Data Loaders
    # Note: get_dataloaders returns fixed split based on metadata files.
    # Ideally, for true K-Fold, the metadata would change, but here we use the provided structure.
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    print(f"Fold {fold}: Training for {epochs} epochs with patience {patience}...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train and Validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
        )

        # Checkpoint State
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": {"lr": Config.LEARNING_RATE, "stages": Config.MODEL_STAGES},
        }

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(state, is_best=True, fold=fold)
            print(
                f"  -> New best validation loss: {best_val_loss:.10f}. Checkpoint saved."
            )
        else:
            patience_counter += 1
            # Save current model (latest)
            save_checkpoint(state, is_best=False, fold=fold)
            print(
                f"  -> Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print(
        f"Training complete. Total time: {total_time:.2f}s. Best Val Loss: {best_val_loss:.10f}"
    )
