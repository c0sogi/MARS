import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.training_utils import apply_mixup, ModelEMA, EarlyStopping
from library.utils import compute_roc_auc


def train_one_epoch(model, ema_model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup and updates the EMA model.

    Args:
        model (torch.nn.Module): The primary model to train.
        ema_model (ModelEMA): The EMA wrapper for the model.
        loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        # Returns mixed images and two sets of labels with lambda
        mixed_images, labels_a, labels_b, lam = apply_mixup(
            images, labels, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_images)

        # Compute Loss with Mixup
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
            outputs, labels_b
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA model
        if ema_model is not None:
            ema_model.update(model)

        # Accumulate loss
        # Assuming reduction='mean', multiply by batch size to get total loss
        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate (typically the EMA model).
        loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    if len(all_targets) > 0:
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        auc_score = compute_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    return avg_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions for the given loader.

    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (torch.utils.data.DataLoader): Data loader.
        device (str): Device to run on.

    Returns:
        np.ndarray: Array of predicted probabilities.
        np.ndarray: Array of IDs (if available in loader).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            # Handle cases where loader returns (images, ids) or just images
            if len(batch) == 2:
                images, ids = batch
                all_ids.extend(ids.numpy())
            else:
                images = batch[0]

            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)
    return all_preds, np.array(all_ids)


def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, save_path
):
    """
    Orchestrates the training process, including EMA, scheduling, and early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device.
        num_epochs (int): Max epochs.
        save_path (str): Path to save the best model.
    """
    # Initialize EMA model
    ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE, mode="max", verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, ema_model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate using the EMA model for better generalization stability
        val_loss, val_auc = validate(
            ema_model.get_model(), val_loader, criterion, device
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Check Early Stopping
        # We pass the ema_model wrapper; EarlyStopping handles saving the inner model
        early_stopping(val_auc, ema_model, save_path)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
