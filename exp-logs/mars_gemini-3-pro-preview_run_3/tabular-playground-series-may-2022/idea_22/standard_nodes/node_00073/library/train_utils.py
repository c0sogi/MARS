import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    """
    Trains the AV-PFE model for one epoch.

    Args:
        model: The AV-PFE model instance.
        loader: DataLoader for the training set.
        optimizer: The optimizer (e.g., AdamW).
        scheduler: The learning rate scheduler (e.g., OneCycleLR).
        device: Torch device (CPU or CUDA).
        criterion: Loss function (expected BCEWithLogitsLoss).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    # Determine number of streams from Config to handle loss summation
    num_streams = len(Config.STREAM_CONFIGS)

    for batch_idx, (x_cat, x_cont, y) in enumerate(loader):
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass: returns (Batch, 5) logits
        outputs = model(x_cat, x_cont)

        # Compute loss: Sum of BCE for each stream
        # This enforces independent learning for each stream
        loss = 0.0
        for i in range(num_streams):
            # Slice to keep dimension (Batch, 1) for broadcasting with y
            stream_output = outputs[:, i : i + 1]
            loss += criterion(stream_output, y)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, loader, device, criterion):
    """
    Evaluates the AV-PFE model on the validation set.

    Args:
        model: The AV-PFE model instance.
        loader: DataLoader for the validation set.
        device: Torch device.
        criterion: Loss function.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = len(loader)

    num_streams = len(Config.STREAM_CONFIGS)

    with torch.no_grad():
        for x_cat, x_cont, y in loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x_cat, x_cont)

            # Compute validation loss (Sum of BCEs)
            loss = 0.0
            for i in range(num_streams):
                loss += criterion(outputs[:, i : i + 1], y)

            total_loss += loss.item()

            # Calculate probabilities for metric
            # 1. Apply Sigmoid to logits -> Probabilities
            # 2. Mean across streams (dim=1) -> Ensemble Prediction
            probs = torch.sigmoid(outputs).mean(dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Concatenate predictions and targets from all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute ROC AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Fallback if only one class is present in the validation set
        auc = 0.5

    # Print metrics with full precision as requested
    print(f"Validation Loss: {avg_loss:.20f}")
    print(f"Validation AUC: {auc:.20f}")

    return avg_loss, auc
