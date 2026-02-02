import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, accuracy_score
from library.config import CFG
from library.utils import AverageMeter, print_metrics


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        dataloader: Training DataLoader.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    losses = AverageMeter()

    # Binary Cross Entropy with Logits for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        # Ensure labels are float and have correct shape [B, 1]
        labels = labels.to(device).unsqueeze(1)

        batch_size = images.size(0)

        # Forward pass
        y_preds = model(images)
        loss = criterion(y_preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)

    # Step the scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    print(f"Epoch {epoch} Training Loss: {losses.avg}")

    return losses.avg


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        device: Device to validate on.

    Returns:
        tuple: (avg_loss, log_loss, accuracy, predictions)
    """
    model.eval()

    losses = AverageMeter()
    preds = []
    targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            batch_size = images.size(0)

            y_preds = model(images)
            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to get probabilities [0, 1]
            preds.append(torch.sigmoid(y_preds).cpu().numpy())
            targets.append(labels.cpu().numpy())

    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate Metrics
    # Log Loss (Primary Metric)
    try:
        val_log_loss = log_loss(targets, predictions)
    except Exception:
        val_log_loss = 0.0

    # Accuracy (Threshold 0.5)
    val_acc = accuracy_score(targets, (predictions > 0.5).astype(int))

    # Print full precision metrics
    metrics = {
        "Validation Loss": losses.avg,
        "Validation LogLoss": val_log_loss,
        "Validation Accuracy": val_acc,
    }
    print_metrics(metrics)

    return losses.avg, val_log_loss, val_acc, predictions


def predict_tta(model, dataloader, device):
    """
    Performs inference with Test Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model.
        dataloader: Test DataLoader (yields image, id).
        device: Device to predict on.

    Returns:
        tuple: (predictions, ids)
    """
    model.eval()

    preds = []
    ids_list = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            images = batch[0].to(device)
            # batch[1] contains IDs for test set
            ids = batch[1]

            # 1. Forward pass with original images
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Forward pass with horizontally flipped images
            # Flip along width dimension (dim 3 for N,C,H,W)
            images_flip = torch.flip(images, dims=[3])
            out_flip = model(images_flip)
            prob_flip = torch.sigmoid(out_flip)

            # 3. Average probabilities
            prob_avg = (prob_orig + prob_flip) / 2.0

            preds.append(prob_avg.cpu().numpy())
            ids_list.append(ids.numpy())

    predictions = np.concatenate(preds)
    ids_result = np.concatenate(ids_list)

    return predictions, ids_result
