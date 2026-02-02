import time
import torch
import torch.nn as nn
from typing import Optional
from timm.data import Mixup

from library.utils import AverageMeter, accuracy, get_logger

logger = get_logger()


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    model_ema: Optional[object] = None,
    mixup_fn: Optional[Mixup] = None,
    accumulation_steps: int = 1,
):
    """
    Trains the model for one epoch.

    Handles:
    1. Dynamic MixUp/CutMix application (active in Phase 1, inactive in Phase 2).
    2. Gradient Accumulation to maintain effective batch size.
    3. EMA updates synchronized with optimizer steps.
    """
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # Ensure gradients are zeroed before starting the loop
    optimizer.zero_grad()

    start_time = time.time()

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply MixUp/CutMix if active (Phase 1)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights and EMA at accumulation boundary
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        # Update metrics
        # We multiply loss by accumulation_steps to log the 'true' loss for this batch
        loss_val = loss.item() * accumulation_steps
        losses.update(loss_val, images.size(0))

        # Compute accuracy only if MixUp is NOT active (Phase 2)
        # With MixUp, targets are soft probabilities, making Top-1 Acc ambiguous/noisy
        if mixup_fn is None:
            acc1 = accuracy(outputs, targets, topk=(1,))
            top1.update(acc1[0].item(), images.size(0))

        # Log progress periodically
        if step % max(1, len(train_loader) // 10) == 0:
            if mixup_fn is not None:
                logger.info(
                    f"Epoch: [{epoch}][{step}/{len(train_loader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
                )
            else:
                logger.info(
                    f"Epoch: [{epoch}][{step}/{len(train_loader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc@1: {top1.val:.2f} ({top1.avg:.2f})"
                )

    end_time = time.time()
    logger.info(
        f"Epoch {epoch} finished. Time: {end_time - start_time:.2f}s. Avg Loss: {losses.avg:.4f}"
    )

    return losses.avg


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Update metrics
            acc1 = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

    # Print full precision metrics as requested
    logger.info(f"Validation Results - Loss: {losses.avg}, Accuracy: {top1.avg}")

    return top1.avg, losses.avg
