import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import Config
from library.utils import set_seed, rmsle
from library.data import get_dataloaders
from library.model import ACC_WDS


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        atomic_features = batch["atomic_features"].to(device)
        global_features = batch["global_features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_features, global_features, mask)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (Average validation loss, RMSLE score)
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic_features, global_features, mask)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)

            # Store predictions and targets for metric calculation
            # Inverse transform log(1+x) -> exp(x) - 1 to get original scale
            preds_original = torch.expm1(outputs)
            targets_original = torch.expm1(targets)

            all_preds.append(preds_original.cpu().numpy())
            all_targets.append(targets_original.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate RMSLE using the utility function
    # Note: The utility function applies log1p internally, so we pass original scale values
    score = rmsle(y_true, y_pred)

    return epoch_loss, score


def run_training(load_cached_data=True):
    """
    Main training loop.
    """
    # Reproducibility
    set_seed(Config.SEED)

    # Device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Model
    model = ACC_WDS().to(device)

    # Optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val RMSLE: {val_rmsle}")

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"New best model saved to {Config.MODEL_CHECKPOINT}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
