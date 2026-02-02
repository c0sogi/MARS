import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training dataloader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to use.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        # Ensure target is (B, 1) to match model output
        target = target.to(device).unsqueeze(1)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation dataloader.
        device (torch.device): The device to use.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))

            # Apply sigmoid to convert logits to probabilities for AUC calculation
            probs = torch.sigmoid(output)

            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate results from all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
    else:
        return 0.0, 0.5

    # Calculate AUC
    # Handle edge case where validation set might only contain one class (e.g., in small debug runs)
    if len(np.unique(all_targets)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_targets, all_probs)

    return losses.avg, auc
