import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, steps_per_epoch=None
):
    """
    Trains the model for one epoch using Mixed Precision.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (torch.device): The device to train on.
        scaler (GradScaler): The gradient scaler for AMP.
        steps_per_epoch (int, optional): Limit number of steps for debugging.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for step, (images, labels) in enumerate(loader):
        if steps_per_epoch is not None and step >= steps_per_epoch:
            break

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device, steps_per_epoch=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion (Loss): The loss function.
        device (torch.device): The device to evaluate on.
        steps_per_epoch (int, optional): Limit number of steps for debugging.

    Returns:
        tuple: (average_loss, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    dataset_size = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            if steps_per_epoch is not None and step >= steps_per_epoch:
                break

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])

    return epoch_loss, preds, targets


def predict(model, loader, device, use_tta=False, steps_per_epoch=None):
    """
    Generates predictions for the test set, optionally using TTA.

    Args:
        model (nn.Module): The model to use for prediction.
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run on.
        use_tta (bool): If True, applies horizontal flip TTA.
        steps_per_epoch (int, optional): Limit number of steps for debugging.

    Returns:
        tuple: (predictions, ids)
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for step, (images, img_ids) in enumerate(loader):
            if steps_per_epoch is not None and step >= steps_per_epoch:
                break

            images = images.to(device)

            # Forward pass 1 (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Forward pass 2 (Horizontal Flip)
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)

                # Average probabilities
                probs = (probs + probs_flip) / 2.0

            preds.append(probs.cpu().numpy())
            ids.extend(img_ids.numpy())

    preds = np.concatenate(preds) if preds else np.array([])
    ids = np.array(ids)
    return preds, ids
