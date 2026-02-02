import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, model_ema, train_loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and updates the EMA model.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        # Reshape labels to (B, 1) to match logits shape
        labels = labels.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad()

        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Update EMA model weights
        if model_ema is not None:
            model_ema.update(model)

        metric_monitor.update("Loss", loss.item())

    # Print average training loss for the epoch
    print(f"Epoch {epoch} Training Loss: {metric_monitor.metrics['Loss']['avg']}")
    return metric_monitor.metrics["Loss"]["avg"]


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    Calculates Loss and ROC AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            metric_monitor.update("Loss", loss.item())

    # Flatten arrays for metric calculation
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        # Handle edge case where only one class is present in the batch/split
        auc = 0.5

    metric_monitor.update("AUC", auc)

    # Print full precision metrics
    print(f"Validation Loss: {metric_monitor.metrics['Loss']['avg']}")
    print(f"Validation AUC: {auc}")

    return metric_monitor.metrics["Loss"]["avg"], auc


def predict_tta(model, test_loader, device):
    """
    Generates predictions using 8-view Test Time Augmentation (TTA).
    Views: Original + 3 Rotations, Horizontal Flip + 3 Rotations.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            # Generate 8 views
            views = []

            # 1. Original and its 3 rotations (0, 90, 180, 270)
            for k in range(4):
                views.append(torch.rot90(images, k, [2, 3]))

            # 2. Flipped (Horizontal) and its 3 rotations
            images_flipped = torch.flip(images, [3])
            for k in range(4):
                views.append(torch.rot90(images_flipped, k, [2, 3]))

            # Stack all views: Shape (8 * B, C, H, W)
            input_tensor = torch.cat(views, dim=0)

            # Forward pass
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)

            # Reshape to (8, B, 1) to separate views
            probs = probs.view(8, batch_size, 1)

            # Average predictions across the 8 views
            avg_probs = torch.mean(probs, dim=0)  # Shape (B, 1)

            all_preds.extend(avg_probs.cpu().numpy().flatten())

    return np.array(all_preds)


def save_submission(ids, probabilities, path):
    """
    Saves the predictions to a CSV file in the required format.
    """
    df = pd.DataFrame({"id": ids, "label": probabilities})
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
