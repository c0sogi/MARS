import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import A2SHN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for imgs, incs, labels in loader:
        imgs = imgs.to(device)
        incs = incs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs, incs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        dataset_size += imgs.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Performs validation on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for imgs, incs, labels in loader:
            imgs = imgs.to(device)
            incs = incs.to(device)
            labels = labels.to(device)

            outputs = model(imgs, incs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            dataset_size += imgs.size(0)

    return running_loss / dataset_size


def train_fold(train_loader, val_loader, epochs=50, lr=2e-4, patience=10, device=None):
    """
    Orchestrates the training loop for a single fold.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate for Adam optimizer.
        patience (int): Patience for early stopping.
        device (torch.device): Device to train on.

    Returns:
        model: The trained model with the best validation weights loaded.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a fresh model for this fold
    model = A2SHN().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler based on validation loss
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
