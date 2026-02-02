import os
import torch
import torch.nn as nn
import numpy as np
from library.utils import quadratic_weighted_kappa


def train_one_epoch(model, loader, optimizer, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision (AMP).

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        device: Calculation device (cuda/cpu).
        scaler: GradScaler for AMP.

    Returns:
        float: Average Mean Squared Error (MSE) loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.MSELoss()

    for _, (images, labels) in enumerate(loader):
        images = images.to(device)
        # Ensure labels are float for MSE regression
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    return running_loss / dataset_size


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: Calculation device.

    Returns:
        tuple: (Average MSE Loss, Quadratic Weighted Kappa Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Collect predictions and targets for Kappa calculation
            preds.extend(outputs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / dataset_size
    kappa = quadratic_weighted_kappa(targets, preds)

    return epoch_loss, kappa


def fit_phase(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    save_path,
    patience=5,
    scheduler=None,
):
    """
    Runs a training phase (loop over epochs) with Early Stopping and Checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        device: Device.
        epochs: Maximum number of epochs.
        save_path: Path to save the best model weights.
        patience: Epochs to wait for improvement before early stopping.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: The best validation Kappa score achieved.
    """
    scaler = torch.cuda.amp.GradScaler()
    best_kappa = -float("inf")
    patience_counter = 0

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_kappa = evaluate(model, val_loader, device)

        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Kappa: {val_kappa}"
        )

        # Save best model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load the best weights before returning
    if os.path.exists(save_path):
        print(f"Loading best model weights from {save_path} (Kappa: {best_kappa})")
        model.load_state_dict(torch.load(save_path))

    return best_kappa


def predict(model, loader, device):
    """
    Generates predictions for a dataset (e.g., test set).

    Args:
        model: The PyTorch model.
        loader: DataLoader.
        device: Device.

    Returns:
        np.array: Flattened array of predicted scores.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle cases where loader returns (image, label) or just image
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy().flatten())

    return np.array(preds)
