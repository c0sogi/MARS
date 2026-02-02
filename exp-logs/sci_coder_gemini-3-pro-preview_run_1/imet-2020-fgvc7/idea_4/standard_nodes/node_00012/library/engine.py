import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, calculate_micro_f1


def train_one_epoch(model, ema_model, optimizer, data_loader, device, epoch):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The model to train.
        ema_model (ModelEMA): The EMA model wrapper (optional).
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (DataLoader): Training data loader.
        device (str): Device to use.
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss.
    """
    model.train()

    # Setup loss function with positive weights
    # pos_weight must be on the same device as targets
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses = AverageMeter()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Label Smoothing
        # For BCE: new_targets = targets * (1 - epsilon) + 0.5 * epsilon
        if Config.LABEL_SMOOTHING > 0:
            targets = (
                targets * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()

        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Update EMA model
        if ema_model is not None:
            ema_model.update(model)

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Training Loss: {losses.avg}")
    return losses.avg


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate (can be EMA module).
        data_loader (DataLoader): Validation data loader.
        device (str): Device to use.

    Returns:
        tuple: (avg_loss, micro_f1, all_logits, all_targets)
    """
    model.eval()

    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses = AverageMeter()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            # Store on CPU to avoid OOM
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Micro F1 Score (default threshold 0.5 for monitoring)
    # Full precision printing is requested
    score = calculate_micro_f1(all_logits, all_targets, threshold=0.5)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Micro F1: {score}")

    return losses.avg, score, all_logits, all_targets


def predict(model, data_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The model to use for prediction.
        data_loader (DataLoader): Test data loader.
        device (str): Device to use.

    Returns:
        tuple: (all_logits, all_ids)
    """
    model.eval()

    all_logits = []
    all_ids = []

    with torch.no_grad():
        for images, ids in data_loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)

            all_logits.append(logits.cpu())
            all_ids.extend(ids)

    all_logits = torch.cat(all_logits, dim=0)

    return all_logits, all_ids
