import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import MetricMonitor


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
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


def train_one_epoch(model, train_loader, optimizer, device, epoch, ema_model=None):
    """
    Handles one epoch of training.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Computation device.
        epoch: Current epoch number.
        ema_model: Optional ModelEMA instance for weight averaging.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Apply Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        optimizer.zero_grad()
        outputs = model(mixed_images)

        # Squeeze outputs to match label shape (B,)
        outputs = outputs.squeeze(1)

        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        if ema_model:
            ema_model.update(model)

        metric_monitor.update("Loss", loss.item())

    print(f"Epoch {epoch} Training: {metric_monitor}")
    return metric_monitor.metrics["Loss"]["avg"]


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        val_loader: DataLoader for validation data.
        device: Computation device.

    Returns:
        float: Area Under the ROC Curve (AUC).
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            metric_monitor.update("Loss", loss.item())

            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(targets, preds)
    metric_monitor.update("AUC", auc)

    print(f"Validation: {metric_monitor}")
    return auc


def predict(model, test_loader, device):
    """
    Performs inference on the test set using 8-view Test Time Augmentation (TTA).

    Args:
        model: The PyTorch model for inference.
        test_loader: DataLoader for test data.
        device: Computation device.

    Returns:
        tuple: (ids, predictions)
            ids: List of image IDs.
            predictions: List of predicted probabilities (averaged over TTA views).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            ids = batch["id"]

            # Initialize accumulator for probabilities
            final_probs = torch.zeros(images.size(0), device=device)

            # 8-view TTA (Dihedral Group D4)
            # Variants: Original and Horizontal Flip
            # Rotations: 0, 90, 180, 270 degrees

            base_variants = [images, torch.flip(images, dims=[3])]

            for variant in base_variants:
                for k in range(4):
                    # Rotate by k * 90 degrees
                    # dims=[2, 3] corresponds to H, W
                    img_rot = torch.rot90(variant, k, dims=[2, 3])

                    logits = model(img_rot)
                    probs = torch.sigmoid(logits).squeeze(1)
                    final_probs += probs

            # Average predictions over 8 views
            final_probs /= 8.0

            all_preds.extend(final_probs.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, all_preds
