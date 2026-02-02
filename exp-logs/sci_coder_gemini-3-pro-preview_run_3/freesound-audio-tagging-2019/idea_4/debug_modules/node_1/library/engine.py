import torch
import torch.nn as nn
import numpy as np
import copy
import os
from library.config import Config
from library.utils import calculate_lwlrap


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the training set.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average Loss, LWLRAP Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for scoring
            preds = torch.sigmoid(outputs)

            # Store predictions and targets for metric calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Calculate LWLRAP
    lwlrap = calculate_lwlrap(all_targets, all_preds)

    return epoch_loss, lwlrap


def fit_model(model, train_loader, val_loader):
    """
    Main training loop with Early Stopping and Best Model Checkpointing.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.

    Returns:
        model: The trained model with the best weights loaded.
    """
    device = Config.DEVICE
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total steps for OneCycleLR
    steps_per_epoch = len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_lwlrap = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_lwlrap = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val LWLRAP: {val_lwlrap}"
        )

        # Early Stopping Logic based on LWLRAP (Maximize)
        if val_lwlrap > best_lwlrap:
            best_lwlrap = val_lwlrap
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save the best model to disk
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Val LWLRAP: {best_lwlrap}")

    return model
