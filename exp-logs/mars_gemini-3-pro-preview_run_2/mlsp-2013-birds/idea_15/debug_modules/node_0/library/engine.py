import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, dataloader, device, pos_weight=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (DataLoader): Training data loader.
        device (str): Device to run on ('cuda' or 'cpu').
        pos_weight (torch.Tensor, optional): Class weights for BCEWithLogitsLoss.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define loss function
    # Ensure pos_weight is on the correct device if provided
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    for _, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Apply Mixup if enabled in Config
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device, pos_weight=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (str): Device to run on.
        pos_weight (torch.Tensor, optional): Class weights for BCEWithLogitsLoss.

    Returns:
        tuple: (average_loss, predictions, targets)
            - average_loss (float): The average validation loss.
            - predictions (np.ndarray): Predicted probabilities (Sigmoid applied).
            - targets (np.ndarray): Ground truth labels.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for _, (images, labels, _) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    return avg_loss, preds, targets
