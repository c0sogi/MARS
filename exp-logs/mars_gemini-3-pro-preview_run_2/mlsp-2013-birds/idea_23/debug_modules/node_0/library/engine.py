import numpy as np
import torch
import torch.nn as nn
from library.config import CFG
from library.utils import get_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(train_loader, model, criterion, optimizer, device, scheduler=None):
    """
    Performs one epoch of training using SAM optimizer and Mixup augmentation.

    Args:
        train_loader: DataLoader for training data.
        model: The neural network model.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        optimizer: SAM optimizer instance.
        device: 'cuda' or 'cpu'.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # --- Mixup Augmentation ---
        if CFG.mixup_alpha > 0:
            lam = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
            index = torch.randperm(batch_size).to(device)
            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_labels = lam * labels + (1 - lam) * labels[index]
        else:
            mixed_images = images
            mixed_labels = labels

        # --- Forward Pass 1 (Current Weights) ---
        y_preds = model(mixed_images)
        loss = criterion(y_preds, mixed_labels)

        # --- Backward Pass 1 ---
        # Compute gradients at the current weight w
        loss.backward()

        # --- SAM Step ---
        # Define closure for the second forward-backward pass at perturbed weights (w + epsilon)
        def closure():
            # SAM requires us to re-compute the loss at the perturbed state.
            # We use the same mixed inputs/targets to ensure consistency.
            output = model(mixed_images)
            loss_adv = criterion(output, mixed_labels)
            loss_adv.backward()
            return loss_adv

        # SAM optimizer handles:
        # 1. Ascent (first_step): w -> w + epsilon
        # 2. Closure call: compute gradients at w + epsilon
        # 3. Descent (second_step): w + epsilon -> w, then update w using gradients from step 2
        optimizer.step(closure)
        optimizer.zero_grad()

        losses.update(loss.item(), batch_size)

        if scheduler is not None:
            scheduler.step()

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    """
    Performs validation on the validation set.

    Args:
        valid_loader: DataLoader for validation data.
        model: The neural network model.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (average_loss, auc_score, predictions)
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Standard forward pass (no Mixup, no SAM)
            y_preds = model(images)
            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)

            # Store predictions (sigmoid applied) and targets for AUC calculation
            preds.append(y_preds.sigmoid().cpu().numpy())
            targets.append(labels.cpu().numpy())

    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate Macro-Averaged ROC AUC
    score = get_score(targets, predictions)

    return losses.avg, score, predictions
