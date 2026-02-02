import torch
import numpy as np
from library.utils import AverageMeter, quadratic_weighted_kappa


def train_one_epoch(
    model, loader, criterion, optimizer, device, accumulation_steps=1, use_amp=True
):
    """
    Trains the model for one epoch using gradient accumulation and mixed precision.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on.
        accumulation_steps: Number of steps to accumulate gradients before updating.
        use_amp: Whether to use Automatic Mixed Precision.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    # Initialize scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Update metrics (scale back up to log the actual loss per batch)
        losses.update(loss.item() * accumulation_steps, images.size(0))

        # Optimizer step
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    print(f"Train Loss: {losses.avg}")
    return losses.avg


def validate(model, loader, criterion, device, use_amp=True):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to evaluate on.
        use_amp: Whether to use Automatic Mixed Precision.

    Returns:
        tuple: (average_loss, qwk_score)
    """
    model.eval()
    losses = AverageMeter()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            preds_list.append(outputs.float().cpu().numpy())
            targets_list.append(targets.float().cpu().numpy())

    preds_arr = np.concatenate(preds_list).flatten()
    targets_arr = np.concatenate(targets_list).flatten()

    # Targets are floats in dataset (for MSE), convert to int for QWK calculation
    targets_int = np.round(targets_arr).astype(int)

    score = quadratic_weighted_kappa(targets_int, preds_arr)

    print(f"Val Loss: {losses.avg}")
    print(f"Val QWK: {score}")

    return losses.avg, score


def inference(model, loader, device, use_amp=True):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        loader: The test DataLoader.
        device: The device to evaluate on.
        use_amp: Whether to use Automatic Mixed Precision.

    Returns:
        np.array: Flattened array of predicted scores.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            # Handle case where loader might return (images, labels) or just images
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)

            preds_list.append(outputs.float().cpu().numpy())

    return np.concatenate(preds_list).flatten()
