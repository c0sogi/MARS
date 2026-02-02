import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger, calculate_weighted_loss


def loss_fn(logits, targets):
    """
    Hierarchical Compound Loss with Label Smoothing.
    Sum of BCEWithLogits for the patient_overall label and
    the mean BCEWithLogits for the 7 vertebral sub-labels.

    Args:
        logits (torch.Tensor): Shape (B, 8). Col 0 is patient_overall, 1-7 are C1-C7.
        targets (torch.Tensor): Shape (B, 8). Same structure.

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Apply mild label smoothing to improve Log Loss calibration
    smoothing = 0.01
    targets = targets * (1.0 - smoothing) + 0.5 * smoothing

    # Patient level loss (Index 0)
    patient_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], targets[:, 0], reduction="mean"
    )

    # Vertebral level loss (Indices 1-7)
    # Averaged across the 7 vertebrae and the batch
    vertebral_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:], targets[:, 1:], reduction="mean"
    )

    # Implicit weighting as per idea description
    return patient_loss + vertebral_loss


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = loss_fn(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, device):
    """
    Performs validation and calculates the competition metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []
    all_study_ids = []

    cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    with torch.no_grad():
        for images, targets, study_ids in loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            logits = model(images)
            loss = loss_fn(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_study_ids.extend(study_ids)

    epoch_loss = running_loss / dataset_size

    # Prepare DataFrames for metric calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    df_pred = pd.DataFrame(all_preds, columns=cols)
    df_pred["StudyInstanceUID"] = all_study_ids

    df_true = pd.DataFrame(all_targets, columns=cols)
    df_true["StudyInstanceUID"] = all_study_ids

    # Calculate weighted log loss
    metric = calculate_weighted_loss(df_true, df_pred)

    return epoch_loss, metric


def fit(model, train_loader, val_loader, device, epochs=Config.EPOCHS):
    """
    Main training routine.
    """
    logger = get_logger("Engine")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Decoupled Cosine Annealing
    # T_max is set to 1.5x epochs to prevent premature decay
    t_max = int(epochs * Config.T_MAX_MULTIPLIER)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    best_metric = float("inf")

    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_metric = validate_one_epoch(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val Metric: {val_metric:.16f}"
        )

        # Save best model based on competition metric (lower is better)
        if val_metric < best_metric:
            best_metric = val_metric
            logger.info(
                f"Validation metric improved. Saving model to {Config.MODEL_SAVE_PATH}"
            )
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    logger.info(f"Training complete. Best Validation Metric: {best_metric:.16f}")
