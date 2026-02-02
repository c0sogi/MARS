import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.utils import AverageMeter
from library.losses import DistillationLoss


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    scheduler=None,
    criterion=None,
    ema_model=None,
):
    """
    Executes one training epoch.

    Args:
        model: Torch model.
        loader: DataLoader.
        optimizer: Torch optimizer.
        device: Torch device.
        epoch: Current epoch index.
        scheduler: LR scheduler (stepped per iteration).
        criterion: Loss function.
        ema_model: Model EMA instance (optional).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()

    # Loop over batches
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Handle soft targets for distillation
        soft_targets = None
        if "soft_target" in batch:
            soft_targets = batch["soft_target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images)

            if isinstance(criterion, DistillationLoss) and soft_targets is not None:
                loss = criterion(logits, soft_targets, targets)
            elif criterion is not None:
                loss = criterion(logits, targets)
            else:
                # Fallback if no criterion provided
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA Update
        if ema_model is not None:
            ema_model.update(model)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


@torch.no_grad()
def validate(model, loader, device, criterion=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: Torch model.
        loader: DataLoader.
        device: Torch device.
        criterion: Loss function.

    Returns:
        Tuple of (average_loss, logits, targets)
    """
    model.eval()
    losses = AverageMeter()
    all_logits = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = 0.0
            if criterion is not None:
                loss = criterion(logits, targets)
                losses.update(loss.item(), images.size(0))

        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)

    return losses.avg, all_logits, all_targets
