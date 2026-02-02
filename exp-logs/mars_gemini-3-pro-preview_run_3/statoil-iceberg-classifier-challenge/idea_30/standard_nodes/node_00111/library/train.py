import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import get_device, set_seed
from library.model import IcebergCNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        optimizer: The optimizer (e.g., AdamW).
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        device: The device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for (imgs, angles), labels in loader:
        imgs = imgs.to(device)
        angles = angles.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(imgs, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (imgs, angles), labels in loader:
            imgs = imgs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(imgs, angles)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


def run_fold(
    train_loader,
    val_loader,
    fold_idx=0,
    epochs=75,
    patience=12,
    lr=1e-3,
    save_dir="./checkpoints",
):
    """
    Runs the training loop for a single fold with early stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        fold_idx: Index of the current fold (for file naming).
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        lr: Learning rate.
        save_dir: Directory to save model checkpoints.

    Returns:
        model: The model with the best validation weights loaded.
    """
    set_seed(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    model = IcebergCNN().to(device)

    # AdamW with Decoupled Weight Decay (Constant LR)
    # This decouples the L2 regularization from the adaptive gradient updates,
    # stabilizing the fusion of raw-scale incidence angles with image features.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f"model_best_fold_{fold_idx}.pth")

    print(f"Starting training for fold {fold_idx} on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Acc: {val_acc}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Fold {fold_idx} training complete. Best Val Loss: {best_loss}")

    # Load best weights to return the optimal model
    model.load_state_dict(torch.load(best_model_path))
    return model
