import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import accuracy

from library.config import Config
from library.utils import MetricMonitor, get_logger
from library.data import get_id_map

logger = get_logger("Engine")


def train_one_epoch(
    model, loader, optimizer, device, epoch, mixup_fn=None, scaler=None
):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        mixup_fn: Mixup function (optional).
        scaler: GradScaler for mixed precision training (optional).

    Returns:
        dict: Averaged metrics for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Select loss function based on Mixup activation
    # If Mixup is active, targets are soft (probabilities), so we use SoftTargetCrossEntropy.
    # If Mixup is inactive, targets are indices, so we use LabelSmoothingCrossEntropy.
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        # Config specifies label_smoothing=0.1 for both phases
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        # If mixup is used, targets are (N, C), take argmax for accuracy calc
        if mixup_fn is not None:
            acc_targets = targets.argmax(dim=1)
        else:
            acc_targets = targets

        (acc1,) = accuracy(outputs, acc_targets, topk=(1,))
        batch_size = images.size(0)

        # Update metrics
        metric_monitor.update("Loss", loss.item(), batch_size)
        metric_monitor.update("Top1_Error", 100.0 - acc1.item(), batch_size)

    logger.info(f"Epoch {epoch} Training: {metric_monitor}")
    return metric_monitor.metrics


def validate(model, loader, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: Device to evaluate on.

    Returns:
        dict: Averaged metrics for the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    # Use LabelSmoothingCrossEntropy for validation loss consistency
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            (acc1,) = accuracy(outputs, targets, topk=(1,))
            batch_size = images.size(0)

            metric_monitor.update("Loss", loss.item(), batch_size)
            metric_monitor.update("Top1_Error", 100.0 - acc1.item(), batch_size)

    logger.info(f"Validation: {metric_monitor}")
    return metric_monitor.metrics


def predict(model, loader, device, output_file=Config.SUBMISSION_FILE):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Implements Test Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model.
        loader: DataLoader for test data.
        device: Device to predict on.
        output_file: Path to save the submission CSV.
    """
    model.eval()

    # Load ID mapping to convert indices back to category_ids
    _, idx2id = get_id_map()

    all_preds = []
    image_ids = loader.dataset.df["image_id"].values

    logger.info("Starting inference with TTA (Horizontal Flip)...")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            # Standard forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            if Config.INFERENCE["tta"]:
                # Horizontal Flip TTA
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = F.softmax(outputs_flipped, dim=1)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            # Get Top-K predictions
            topk_vals, topk_indices = torch.topk(
                probs, k=Config.INFERENCE["top_k"], dim=1
            )
            all_preds.append(topk_indices.cpu().numpy())

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Format for submission
    logger.info("Formatting submission...")
    submission_rows = []

    for i, img_id in enumerate(image_ids):
        indices = all_preds[i]
        # Map indices to category IDs
        cat_ids = [str(idx2id[idx]) for idx in indices]
        predicted_str = " ".join(cat_ids)

        submission_rows.append({"id": img_id, "predicted": predicted_str})

    # Create DataFrame and Save
    submission_df = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    logger.info(f"Submission saved to {output_file}")
