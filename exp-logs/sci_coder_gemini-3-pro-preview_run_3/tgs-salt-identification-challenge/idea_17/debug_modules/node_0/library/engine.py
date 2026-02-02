import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from library.config import Config
from library.utils import calc_map


def train_one_epoch(
    model, loader, optimizer, scaler, loss_fn, device, deep_supervision=True
):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        scaler: GradScaler for AMP.
        loss_fn: Loss function module.
        device: 'cuda' or 'cpu'.
        deep_supervision (bool): Whether to use deep supervision (pass list of outputs to loss)
                                 or just the final output.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)

            # Handle Deep Supervision logic
            # If deep_supervision is False, we force the use of only the final head
            # even if the model returns a list.
            if not deep_supervision and isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]

            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, loader, loss_fn, device, deep_supervision=False):
    """
    Evaluates the model on the validation set.
    Performs center cropping to match original 101x101 resolution before metric calculation.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        loss_fn: Loss function module.
        device: 'cuda' or 'cpu'.
        deep_supervision (bool): Whether to calculate loss on all heads or just the final one.
                                 Usually False for validation.

    Returns:
        tuple: (average_loss, map_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # Calculate crop indices to revert 128x128 padding to 101x101
    # Albumentations PadIfNeeded with center default:
    # (128 - 101) / 2 = 13.5 -> 13 pixels on top/left
    start_idx = (Config.MODEL_HEIGHT - Config.ORIG_HEIGHT) // 2
    end_idx = start_idx + Config.ORIG_HEIGHT

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.size(0)

            with autocast():
                outputs = model(images)

            # For validation/metrics, we primarily care about the final prediction
            # But we calculate loss based on the requested mode
            loss_input = outputs
            if not deep_supervision and isinstance(outputs, (list, tuple)):
                loss_input = outputs[-1]

            loss = loss_fn(loss_input, masks)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Prepare predictions for mAP calculation
            # 1. Extract final output if list
            if isinstance(outputs, (list, tuple)):
                final_output = outputs[-1]
            else:
                final_output = outputs

            # 2. Apply Sigmoid to get probabilities
            probs = torch.sigmoid(final_output)

            # 3. Crop to original resolution (101x101)
            # Shape: (B, 1, H, W) or (B, H, W)
            if probs.ndim == 4:
                probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]
                masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]
            else:
                probs_cropped = probs[:, start_idx:end_idx, start_idx:end_idx]
                masks_cropped = masks[:, start_idx:end_idx, start_idx:end_idx]

            # 4. Accumulate on CPU to avoid GPU OOM
            all_preds.append(probs_cropped.cpu())
            all_targets.append(masks_cropped.cpu())

    # Aggregate results
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate mAP using the utility function
        # We use a default probability threshold of 0.5 for binarization in this check
        map_score = calc_map(all_preds, all_targets, threshold=0.5)
    else:
        map_score = 0.0

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    return avg_loss, map_score
