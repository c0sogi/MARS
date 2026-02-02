import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda
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
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup regularization.
    """
    losses = AverageMeter()
    model.train()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        # Ensure targets are (N, 1) for BCEWithLogitsLoss
        targets = targets.to(device).view(-1, 1)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Training Loss: {losses.avg}")
    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    losses = AverageMeter()
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    auc = calculate_roc_auc(all_targets, all_preds)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {auc}")

    return losses.avg, auc


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Original + HFlip + VFlip).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            img_hflip = torch.flip(images, [3])
            out_hflip = model(img_hflip)
            prob_hflip = torch.sigmoid(out_hflip)

            # 3. Vertical Flip (dim 2 is height)
            img_vflip = torch.flip(images, [2])
            out_vflip = model(img_vflip)
            prob_vflip = torch.sigmoid(out_vflip)

            # Average probabilities
            avg_prob = (prob_orig + prob_hflip + prob_vflip) / 3.0
            all_preds.append(avg_prob.cpu().numpy())

    return np.concatenate(all_preds)
