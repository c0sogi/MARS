import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import MetricMonitor, calculate_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns:
        mixed_x: Mixed inputs
        y_a: Targets for the first image
        y_b: Targets for the second image
        lam: Lambda mixing coefficient
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
    Computes the Mixup loss (weighted average of losses).
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, train_loader, optimizer, device, epoch):
    """
    Training loop for one epoch with Mixup regularization.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup
        if Config.MIXUP_ALPHA > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.MIXUP_ALPHA, device
            )

        optimizer.zero_grad()

        outputs = model(images)

        # Calculate Loss
        if Config.MIXUP_ALPHA > 0:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    print(f"Epoch {epoch} Train: {metric_monitor}")
    return metric_monitor.get_avg("Loss")


def valid_epoch(model, val_loader, device):
    """
    Validation loop. Evaluates model and calculates AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for images, label in val_loader:
            images = images.to(device)
            label = label.to(device)

            outputs = model(images)
            loss = criterion(outputs, label)

            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid to get probabilities
            prob = torch.sigmoid(outputs)
            preds.extend(prob.cpu().numpy())
            targets.extend(label.cpu().numpy())

    # Calculate AUC
    # Flatten arrays to ensure correct shape
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    auc = calculate_auc(targets, preds)
    print(f"Validation: {metric_monitor} | AUC: {auc:.6f}")

    return metric_monitor.get_avg("Loss"), auc


def inference_fn(model, test_loader, device):
    """
    Inference loop with 8-view Test Time Augmentation (TTA).
    Averages predictions across 8 dihedral symmetries (rotations and flips).
    """
    model.eval()
    preds = []

    # Define 8 Dihedral transforms (Rotations and Flips)
    # These cover the symmetries of a square
    transforms = [
        lambda x: x,  # Identity
        lambda x: torch.rot90(x, 1, (2, 3)),  # Rot 90
        lambda x: torch.rot90(x, 2, (2, 3)),  # Rot 180
        lambda x: torch.rot90(x, 3, (2, 3)),  # Rot 270
        lambda x: torch.flip(x, (3,)),  # Flip Horizontal
        lambda x: torch.rot90(
            torch.flip(x, (3,)), 1, (2, 3)
        ),  # Flip H + Rot 90 (Transpose)
        lambda x: torch.rot90(
            torch.flip(x, (3,)), 2, (2, 3)
        ),  # Flip H + Rot 180 (Flip Vertical)
        lambda x: torch.rot90(
            torch.flip(x, (3,)), 3, (2, 3)
        ),  # Flip H + Rot 270 (Anti-Transpose)
    ]

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.shape[0]

            # Accumulator for predictions
            batch_preds = torch.zeros((batch_size, 1), device=device)

            # Apply each transform, predict, and accumulate
            for t in transforms:
                aug_images = t(images)
                outputs = model(aug_images)
                batch_preds += torch.sigmoid(outputs)

            # Average predictions across all views
            batch_preds /= len(transforms)
            preds.extend(batch_preds.cpu().numpy())

    # Return flattened probabilities
    return np.concatenate(preds).flatten()
