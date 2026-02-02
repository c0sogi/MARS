import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import EarlyStopping
from library.model import CADPNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Calculation device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for images, angles, labels in dataloader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / dataset_size


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Calculation device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for images, angles, labels in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / dataset_size


def train_fold(fold_idx, train_loader, val_loader, device):
    """
    Orchestrates the training process for a single fold, including
    initialization, optimization, scheduling, and early stopping.

    Args:
        fold_idx (int): Index of the current fold (0-based).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Calculation device.

    Returns:
        model: The model with the best weights loaded.
    """
    print(f"\nStarting training for Fold {fold_idx + 1}")

    # Initialize Model
    model = CADPNet().to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Early Stopping
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=checkpoint_path
    )

    # Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model weights before returning
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def predict(model, dataloader, device):
    """
    Generates probability predictions for a given dataloader.

    Args:
        model: Trained PyTorch model.
        dataloader: Test dataloader.
        device: Calculation device.

    Returns:
        np.array: Predicted probabilities of shape (N, 1).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, angles in dataloader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.vstack(preds)
