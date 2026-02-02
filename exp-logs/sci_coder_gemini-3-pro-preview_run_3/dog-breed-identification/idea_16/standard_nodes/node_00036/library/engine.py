import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import get_logger, calculate_score

# Initialize logger
logger = get_logger("engine", os.path.join(Config.working_dir, "engine.log"))


def train_head_only(model, data_loader, optimizer, device, epoch):
    """
    Executes the 'Head Warmup' phase (Phase 1) where the backbone is frozen
    and only the classification head is trained with a high learning rate.

    Args:
        model (DogClassifier): The model instance.
        data_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer configured for the head.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    # Enforce backbone freezing
    model.freeze_backbone()

    total_loss = 0.0
    num_batches = len(data_loader)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} [Head Warmup] - Train Loss: {avg_loss}")

    return avg_loss


def train_one_epoch(model, data_loader, optimizer, device, epoch, scheduler=None):
    """
    Executes the 'Full Fine-tuning' phase (Phase 2) where the entire network
    is unfrozen and trained.

    Args:
        model (DogClassifier): The model instance.
        data_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer configured for the full network.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        scheduler (optional): Learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    # Enforce unfreezing of all parameters
    model.unfreeze_all()

    total_loss = 0.0
    num_batches = len(data_loader)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step scheduler if it is batch-level (e.g., OneCycleLR), though Config implies Cosine
        # We assume standard epoch-based stepping in the main loop, but if a scheduler is passed
        # and requires batch stepping, it can be handled here. For this implementation,
        # we strictly follow the standard loop.

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} [Fine-tuning] - Train Loss: {avg_loss}")

    return avg_loss


def validate(model, data_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (DogClassifier): The model instance.
        data_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        device (torch.device): Device to run evaluation on.

    Returns:
        float: The calculated log loss (metric).
        np.ndarray: The predicted probabilities.
    """
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass (Logits)
            logits = model(images)

            # Convert logits to probabilities for Log Loss metric
            probs = torch.softmax(logits, dim=1)

            final_outputs.append(probs.cpu().numpy())
            final_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    y_pred = np.vstack(final_outputs)
    y_true = np.concatenate(final_targets)

    # Calculate Multi Class Log Loss using the utility function
    # Note: calculate_score wraps sklearn.metrics.log_loss
    metric_score = calculate_score(y_true, y_pred)

    logger.info(f"Validation Log Loss: {metric_score}")

    return metric_score, y_pred
