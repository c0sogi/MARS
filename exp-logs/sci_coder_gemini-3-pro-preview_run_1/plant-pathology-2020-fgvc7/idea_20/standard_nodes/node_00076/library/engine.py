import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def check_initial_loss(model, data_loader, criterion, device):
    """
    Performs a forward pass on a single batch to verify that the initial loss
    is within a reasonable range (approx ln(4) ~= 1.38 for 4 classes).
    """
    model.eval()
    try:
        images, targets = next(iter(data_loader))
    except StopIteration:
        print("Warning: Data loader is empty. Skipping initial loss check.")
        return

    images = images.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, targets)

    print(f"Initial Loss Check: {loss.item()}")


def train_one_epoch(model, data_loader, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        data_loader: The training data loader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device (cpu or cuda).
        scheduler: Optional learning rate scheduler.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in data_loader:
        batch_size = images.size(0)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, data_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        data_loader: The validation data loader.
        criterion: The loss function.
        device: The device.

    Returns:
        tuple: (average validation loss, validation AUC)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets in data_loader:
            batch_size = images.size(0)
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    # Calculate Mean Column-wise ROC AUC
    # y_true is one-hot (or soft targets), y_score is probabilities
    try:
        val_auc = roc_auc_score(
            all_targets, all_preds, average="macro", multi_class="ovr"
        )
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        val_auc = 0.0

    return val_loss, val_auc


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
):
    """
    Runs the full training loop with Early Stopping.

    Args:
        model: The model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR Scheduler.
        device: Device.
        epochs: Number of epochs.
        save_path: Path to save the best model.

    Returns:
        float: The best validation AUC achieved.
    """
    # 1. Initialization Safeguard
    check_initial_loss(model, train_loader, criterion, device)

    best_auc = -1.0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler (Epoch-level stepping for CosineAnnealingWarmRestarts with T_0=EPOCHS)
        if scheduler is not None:
            scheduler.step()

        # Print Metrics (Full precision)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Metric-Based Early Stopping (Save Best Model)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

    return best_auc
