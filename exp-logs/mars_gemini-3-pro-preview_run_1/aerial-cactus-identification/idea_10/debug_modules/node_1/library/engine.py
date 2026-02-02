import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, calculate_roc_auc
from library.config import Config


def train_one_epoch(model, loader, optimizer, device, criterion, mixup_fn=None):
    """
    Trains the model for one epoch using the provided loader and optimizer.
    Applies Mixup regularization if mixup_fn is provided.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Compute device (CPU or CUDA).
        criterion (nn.Module): Loss function (typically BCEWithLogitsLoss).
        mixup_fn (Mixup, optional): Mixup augmentation object.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, file_sizes, targets) in enumerate(loader):
        images = images.to(device)
        file_sizes = file_sizes.to(device)
        targets = targets.to(device).view(-1, 1)
        # file_sizes are available but not used by the base CNNs directly in Stage 1

        if mixup_fn is not None:
            # Mixup returns: mixed_images, mixed_file_sizes, targets_a, targets_b, lam
            images, _, targets_a, targets_b, lam = mixup_fn(
                (images, file_sizes, targets)
            )
            targets_a = targets_a.view(-1, 1)
            targets_b = targets_b.view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        if mixup_fn is not None:
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(
                logits, targets_b
            )
        else:
            loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.

    Returns:
        tuple: (average_loss, auc_score, predictions, targets)
               predictions are probabilities (after sigmoid).
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, file_sizes, targets in loader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            logits = model(images)
            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc, np.array(all_preds), np.array(all_targets)


def predict_tta(model, loader, device):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, 180-degree Rotation.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Compute device.

    Returns:
        np.array: Aggregated probability predictions.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, file_sizes in loader:
            images = images.to(device)

            # 1. Original View
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            logits_2 = model(images_h)
            probs_2 = torch.sigmoid(logits_2)

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            logits_3 = model(images_v)
            probs_3 = torch.sigmoid(logits_3)

            # 4. 180-degree Rotation (Horizontal + Vertical Flip)
            images_hv = torch.flip(images, dims=[2, 3])
            logits_4 = model(images_hv)
            probs_4 = torch.sigmoid(logits_4)

            # Average probabilities across all views
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())

    return np.array(all_preds)
