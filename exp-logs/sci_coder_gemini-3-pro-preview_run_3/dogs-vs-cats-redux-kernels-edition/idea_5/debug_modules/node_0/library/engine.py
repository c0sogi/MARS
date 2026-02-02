import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from library.config import Config


def train_one_epoch(model, optimizer, data_loader, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The training data loader.
        device (str): The device to train on ('cuda' or 'cpu').
        scaler (torch.cuda.amp.GradScaler): The scaler for mixed precision training.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for images, targets in data_loader:
        images = images.to(device)
        # Targets from dataset are [Batch], model output is [Batch, 1]
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate Loss
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The validation data loader.
        device (str): The device to evaluate on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Convert logits to probabilities for accuracy calculation
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Accuracy (Threshold = 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, preds_binary)

    return avg_loss, accuracy
