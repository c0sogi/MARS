import os
import numpy as np
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from library.utils import get_device, seed_everything
from library.dataset import IcebergDataset, get_transforms
from library.model import APCNN, get_optimizer, get_criterion


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (torch.device): Device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, log_loss_score)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            target_tensor = targets.to(device).view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, target_tensor)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Apply sigmoid to get probabilities for log_loss
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    avg_loss = running_loss / count

    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.concatenate(all_targets)

    # Calculate Log Loss (clipping is handled internally by sklearn usually,
    # but strictly speaking log_loss handles 0/1 inputs fine with probabilities)
    score = log_loss(y_true, y_pred, labels=[0, 1])

    return avg_loss, score


def fit_fold(
    fold,
    X_train,
    angles_train,
    y_train,
    X_val,
    angles_val,
    y_val,
    epochs=50,
    batch_size=32,
    patience=10,
    lr=1e-3,
    weight_decay=1e-4,
    save_dir="./working/idea_18",
):
    """
    Trains a model for a specific fold with early stopping.

    Args:
        fold (int): Fold index.
        X_train, angles_train, y_train: Training data.
        X_val, angles_val, y_val: Validation data.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size.
        patience (int): Early stopping patience.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization factor.
        save_dir (str): Directory to save checkpoints.

    Returns:
        float: Best validation log loss score.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    # Prepare Datasets and Loaders
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=get_transforms("train"), mode="train"
    )
    val_dataset = IcebergDataset(
        X_val, angles_val, y_val, transform=get_transforms("val"), mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model, Optimizer, Criterion
    model = APCNN().to(device)
    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)
    criterion = get_criterion()

    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f"model_fold_{fold}.pth")

    print(f"Starting training for Fold {fold}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val LogLoss: {val_score}"
        )

        # Early Stopping Logic
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved for Fold {fold} with LogLoss: {best_score}")
        else:
            patience_counter += 1
            # print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Fold {fold} finished. Best Validation LogLoss: {best_score}")
    return best_score
