import os
import numpy as np
import torch
import torch.nn as nn
from library.utils import calculate_roc_auc


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter.
        device (str): Device to perform operations on.

    Returns:
        tuple: (mixed_x, mixed_y)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # For multi-label BCE, we can mix the targets directly
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup regularization.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup with alpha=0.4 as per strategy
        mixed_images, mixed_labels = mixup_data(
            images, labels, alpha=0.4, device=device
        )

        optimizer.zero_grad()

        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def validate_one_epoch(model, loader, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: Device to evaluate on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        auc_score = calculate_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.5

    # Print full precision metrics
    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation AUC: {auc_score}")

    return epoch_loss, auc_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training loop with Early Stopping and Checkpointing.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model checkpoint.

    Returns:
        float: Best AUC achieved.
    """
    best_auc = 0.0
    patience_counter = 0

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"--- Epoch {epoch}/{num_epochs} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc = validate_one_epoch(model, val_loader, device)

        if scheduler:
            # Handle different scheduler types if necessary, assuming standard step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best AUC: {best_auc}")
    return best_auc
