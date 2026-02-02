import torch
import numpy as np
from library.utils import calculate_roc_auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The training data loader.
        criterion (torch.nn.modules.loss._Loss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.

    Returns:
        tuple: (average_loss, average_auc)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)

        # CrossEntropyLoss expects class indices, but targets are one-hot/probabilities
        # We use argmax to get the class index
        loss = criterion(outputs, torch.argmax(targets, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Store predictions and targets for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        # Apply softmax to get probabilities for AUC
        all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        epoch_auc = calculate_roc_auc(all_targets, all_preds)
    else:
        epoch_auc = 0.0

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.modules.loss._Loss): The loss function.
        device (torch.device): The device to use for validation.

    Returns:
        tuple: (average_loss, average_auc)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    dataset_size = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, torch.argmax(targets, dim=1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        val_auc = calculate_roc_auc(all_targets, all_preds)
    else:
        val_auc = 0.0

    return val_loss, val_auc
