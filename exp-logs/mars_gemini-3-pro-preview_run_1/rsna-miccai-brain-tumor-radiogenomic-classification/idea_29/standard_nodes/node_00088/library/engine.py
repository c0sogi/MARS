import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import AverageMeter


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    # Set model to training mode (enables Dropout and Batch Norm updates)
    model.train()

    losses = AverageMeter("Train Loss")

    # Iterate over the training data
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move data to the defined device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(
            1
        )  # Ensure shape matches logits

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Note: SDWIVNet applies Structured Depth Dropout internally during training
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    # Log metrics for the epoch
    logger.info(f"Epoch {epoch} Training Loss: {losses.avg}")

    return losses.avg


def validate(val_loader, model, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    """
    # Set model to evaluation mode (disables Dropout)
    model.eval()

    losses = AverageMeter("Val Loss")

    # Lists to store predictions and true labels for AUC calculation
    all_targets = []
    all_probs = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities (0-1 range)
            probs = torch.sigmoid(outputs)

            # Move to CPU and store
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Compute ROC AUC
    # Handle edge case where only one class is present in the batch (though unlikely in full val set)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5
        logger.warning(
            "ROC AUC could not be computed (likely only one class in validation set). Defaulting to 0.5."
        )

    # Log metrics with full precision
    logger.info(f"Validation Loss: {losses.avg}")
    logger.info(f"Validation AUC: {auc}")

    return losses.avg, auc
