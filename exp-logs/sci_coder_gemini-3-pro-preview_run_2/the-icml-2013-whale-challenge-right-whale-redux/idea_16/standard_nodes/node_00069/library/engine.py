import torch
import numpy as np
from library.utils import calculate_auc


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): The optimizer to update model weights.
        device (torch.device): The device (CPU/GPU) to run the computations on.
        criterion (torch.nn.Module): The loss function (e.g., BCEWithLogitsLoss).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)  # Ensure targets are (Batch, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): The device (CPU/GPU) to run the computations on.
        criterion (torch.nn.Module): The loss function.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average validation loss.
            - auc_score (float): The Area Under the ROC Curve score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Store predictions and targets for AUC calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate and flatten arrays for metric calculation
    all_targets = np.concatenate(all_targets).ravel()
    all_preds = np.concatenate(all_preds).ravel()

    # Calculate AUC
    auc_score = calculate_auc(all_targets, all_preds)

    return epoch_loss, auc_score
