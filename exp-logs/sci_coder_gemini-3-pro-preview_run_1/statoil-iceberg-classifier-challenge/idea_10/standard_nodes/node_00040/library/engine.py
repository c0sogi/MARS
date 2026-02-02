import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_log_loss


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using BCEWithLogitsLoss and Label Smoothing.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to train on (cpu or cuda).
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Standard BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Label Smoothing parameter
    epsilon = Config.LABEL_SMOOTHING

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).view(-1, 1)

        batch_size = images.size(0)

        # Apply Label Smoothing manually
        # y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        labels_smoothed = labels * (1.0 - epsilon) + 0.5 * epsilon

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, labels_smoothed)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Validation data loader.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, log_loss_metric)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    labels_list = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).view(-1, 1)

            batch_size = images.size(0)

            logits = model(images, angles)

            # Calculate loss on original labels (no smoothing for validation metric)
            loss = criterion(logits, labels)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(preds_list) > 0:
        preds_all = np.concatenate(preds_list)
        labels_all = np.concatenate(labels_list)

        # Calculate Log Loss
        metric = calculate_log_loss(labels_all, preds_all)
    else:
        metric = 0.0

    return epoch_loss, metric


def predict_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Test data loader.
        device (str): Device to predict on.

    Returns:
        np.array: Flattened array of predicted probabilities.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original View
            logits1 = model(images, angles)
            prob1 = torch.sigmoid(logits1)

            # 2. Horizontal Flip (Width is dim 3: B, C, H, W)
            images_h = torch.flip(images, dims=[3])
            logits2 = model(images_h, angles)
            prob2 = torch.sigmoid(logits2)

            # 3. Vertical Flip (Height is dim 2: B, C, H, W)
            images_v = torch.flip(images, dims=[2])
            logits3 = model(images_v, angles)
            prob3 = torch.sigmoid(logits3)

            # Average the probabilities
            avg_prob = (prob1 + prob2 + prob3) / 3.0

            preds_list.append(avg_prob.cpu().numpy())

    if len(preds_list) > 0:
        return np.concatenate(preds_list).flatten()
    else:
        return np.array([])
