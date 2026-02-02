import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import get_device, BHA_ResNet


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The BHA_ResNet model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer (AdamW).
        criterion (Loss): The loss function (BCEWithLogitsLoss).
        device (torch.device): Computation device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape is (B, 1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The BHA_ResNet model.
        loader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (torch.device): Computation device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return running_loss / total_samples


def fit_fold(
    fold_idx,
    train_loader,
    val_loader,
    epochs=75,
    patience=12,
    lr=1e-3,
    checkpoint_dir="./checkpoints",
):
    """
    Trains the model for a specific fold with Early Stopping.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        lr (float): Learning rate.
        checkpoint_dir (str): Directory to save model checkpoints.

    Returns:
        tuple: (best_model_path, best_val_loss)
    """
    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the BHA-ResNet model
    model = BHA_ResNet().to(device)

    # Optimizer: AdamW with constant learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Loss Function: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx}.pth")

    print(f"--- Fold {fold_idx} Start ---")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Check Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} Best Val Loss: {best_loss:.10f}")

    return best_model_path, best_loss
