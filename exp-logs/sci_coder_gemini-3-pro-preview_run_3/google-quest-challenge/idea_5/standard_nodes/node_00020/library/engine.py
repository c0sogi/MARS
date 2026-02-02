import torch
import numpy as np
from library.utils import compute_spearman_metric


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): The learning rate scheduler.
        device (torch.device): The device to run training on.
        criterion (nn.Module): The loss function (e.g., BCEWithLogitsLoss).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        # Move batch inputs to the computation device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        segment_mask = batch["segment_mask"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model returns (logits, features)
        logits, _ = model(input_ids, attention_mask, segment_mask)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to run evaluation on.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: (average_loss, spearman_score)
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_mask = batch["segment_mask"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            logits, _ = model(input_ids, attention_mask, segment_mask)

            # Compute loss
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities [0, 1]
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute Spearman Correlation
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        spearman_score = compute_spearman_metric(all_targets, all_preds)
    else:
        spearman_score = 0.0

    return avg_loss, spearman_score


def extract_all_features(model, dataloader, device):
    """
    Runs inference to extract features from the model's backbone.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: Array of extracted features with shape (num_samples, feature_dim).
    """
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_mask = batch["segment_mask"].to(device)

            # Forward pass
            # We only care about the features (second return value)
            _, features = model(input_ids, attention_mask, segment_mask)

            all_features.append(features.cpu().numpy())

    if len(all_features) > 0:
        return np.vstack(all_features)
    else:
        return np.array([])
