import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, calculate_log_mae


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function (e.g., L1Loss).
        device (str/torch.device): Computation device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for data in loader:
        data = data.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # The model accepts the PyG Data object directly
        preds = model(data)

        # Calculate loss
        # data.target_val contains the normalized scalar coupling constants
        loss = criterion(preds, data.target_val)

        # Backward pass
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Update loss tracking
        loss_meter.update(loss.item(), n=preds.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, loader, device, stats):
    """
    Evaluates the model on the validation set and calculates the competition metric.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Validation data loader.
        device (str/torch.device): Computation device.
        stats (dict): Dictionary of (mean, std) for each coupling type for denormalization.

    Returns:
        float: The Log Mean Absolute Error (LMAE) metric.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_types = []

    for data in loader:
        data = data.to(device)

        # Forward pass
        preds = model(data)

        # Collect predictions, targets, and types
        all_preds.append(preds)
        all_targets.append(data.target_val)
        all_types.append(data.target_type)

    # Concatenate results from all batches
    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_types = torch.cat(all_types)

    # Calculate Log Mean Absolute Error
    # The utility function handles denormalization using the provided stats
    metric = calculate_log_mae(all_preds, all_targets, all_types, stats)

    return metric.item()
