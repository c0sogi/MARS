import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as TF

from library.config import Config
from library.utils import MetricMonitor


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
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
    Calculates the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and AdamW.
    """
    model.train()
    metric_monitor = MetricMonitor()
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images = images.to(device)
        # Ensure targets are shaped correctly (N, 1) for BCEWithLogitsLoss
        targets = targets.to(device).view(-1, 1)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        metric_monitor.update("loss", loss.item())

    return metric_monitor.get_avg("loss")


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            metric_monitor.update("loss", loss.item())

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate AUC
    # Flatten arrays
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case where only one class is present in batch (unlikely in full val)
        auc = 0.5

    return metric_monitor.get_avg("loss"), auc


def predict_tta(model, test_loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.

    Returns:
        dict: mapping {image_id: probability}
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip
            images_h = TF.hflip(images)
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip
            images_v = TF.vflip(images)
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_probs = (prob_orig + prob_h + prob_v) / 3.0

            # Store results
            avg_probs_np = avg_probs.cpu().numpy().flatten()

            for img_id, prob in zip(ids, avg_probs_np):
                predictions[img_id] = float(prob)

    return predictions
