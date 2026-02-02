import torch
import torch.nn as nn
import numpy as np
from timm.utils import ModelEmaV2
from library.config import Config
from library.utils import AverageMeter, get_score, get_logger

# Initialize logger
logger = get_logger(name="engine")


def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, ema_model=None
):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        dataloader: Training dataloader.
        device: Device to run training on.
        epoch: Current epoch number.
        ema_model: ModelEmaV2 instance for exponential moving average.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    losses = AverageMeter()

    # BCEWithLogitsLoss is standard for multi-label classification
    # pos_weight can be used if defined in Config, but usually handled carefully
    criterion = nn.BCEWithLogitsLoss()

    # Loop over batches
    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Scheduler step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Update EMA
        if ema_model is not None:
            ema_model.update(model)

        # Update metrics
        losses.update(loss.item(), batch_size)

    return losses.avg


@torch.no_grad()
def validate(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model (or EMA model).
        dataloader: Validation dataloader.
        device: Device to run validation on.

    Returns:
        tuple: (average_loss, average_auc_score)
    """
    model.eval()

    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds_list = []
    targets_list = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), batch_size)

        # Apply sigmoid to get probabilities
        preds = torch.sigmoid(outputs)

        preds_list.append(preds.cpu().numpy())
        targets_list.append(targets.cpu().numpy())

    # Concatenate all batches
    preds_arr = np.concatenate(preds_list, axis=0)
    targets_arr = np.concatenate(targets_list, axis=0)

    # Calculate metric
    score = get_score(targets_arr, preds_arr)

    return losses.avg, score
