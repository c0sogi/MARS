import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, compute_roc_auc
from library.data import mixup_data


def train_one_epoch(
    model, loader, optimizer, device, epoch, mixup_alpha=0.0, scheduler=None
):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to run training on ('cuda' or 'cpu').
        epoch (int): Current epoch number (for logging).
        mixup_alpha (float): Alpha parameter for Mixup. If > 0, mixup is applied.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
                                                                      If provided and step-based, stepped here.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha, device
            )
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                outputs, labels_b
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            # Assuming scheduler is stepped per epoch in the main loop,
            # but if it were OneCycleLR it would be stepped here.
            # For this implementation, we assume standard epoch-based scheduling outside,
            # or the user handles it. If it's a batch scheduler:
            # scheduler.step()
            pass

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on a validation or test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for validation/test data.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (avg_loss, roc_auc_score, probabilities, targets)
               probabilities and targets are numpy arrays.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    if len(all_preds) > 0:
        probabilities = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
    else:
        probabilities = np.array([])
        targets = np.array([])

    # Compute Metric
    score = 0.0
    if len(targets) > 0:
        score = compute_roc_auc(targets, probabilities)

    # Print full precision metric as requested
    print(f"Validation Loss: {losses.avg:.16f} | ROC AUC: {score:.16f}")

    return losses.avg, score, probabilities, targets
