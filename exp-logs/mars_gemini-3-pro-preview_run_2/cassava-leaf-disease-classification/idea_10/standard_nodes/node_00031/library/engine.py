import math
import sys
import torch
import torch.nn as nn
from typing import Iterable, Optional, Dict, List
from timm.utils import accuracy
from library.config import Config
from library.utils import AverageMeter, get_logger


def train_one_epoch(
    model: torch.nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_ema: Optional[object] = None,
    mixup_fn: Optional[object] = None,
    scheduler: Optional[object] = None,
) -> Dict[str, float]:
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model to train.
        criterion: The loss function.
        data_loader: Iterable data loader.
        optimizer: The optimizer.
        device: Computation device.
        epoch: Current epoch index.
        model_ema: Optional EMA model wrapper.
        mixup_fn: Optional MixUp/CutMix function.
        scheduler: Optional learning rate scheduler.

    Returns:
        Dict containing average training loss.
    """
    model.train()

    loss_meter = AverageMeter()
    logger = get_logger("train.log")

    # Ensure gradients are zero before starting
    optimizer.zero_grad()

    num_steps = len(data_loader)

    for step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply MixUp / CutMix if enabled for this phase
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # Check for NaN/Inf loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Normalize loss for gradient accumulation
        loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        loss.backward()

        # Optimizer Step (with Gradient Accumulation)
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            # Clip gradients
            if Config.MAX_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            optimizer.zero_grad()

            # Update EMA model
            if model_ema is not None:
                model_ema.update(model)

            # Step scheduler if it requires per-step updates (common in timm)
            if scheduler is not None and hasattr(scheduler, "step_update"):
                num_updates = epoch * num_steps + step
                scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        # Update metrics (using the unscaled loss value)
        loss_meter.update(loss_value, samples.size(0))

        # Log progress
        if step % 100 == 0 or step == num_steps - 1:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch: [{epoch}][{step}/{num_steps}] "
                f"Loss: {loss_meter.val:.6f} ({loss_meter.avg:.6f}) "
                f"LR: {lr:.6f}"
            )

    return {"train_loss": loss_meter.avg}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        criterion: The loss function.
        data_loader: Validation data loader.
        device: Computation device.

    Returns:
        Dict containing validation loss and accuracy.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    logger = get_logger("train.log")

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = samples.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)

    # Log full precision metrics as requested
    logger.info(
        f"Validation Results - "
        f"Loss: {loss_meter.avg} "
        f"Acc@1: {acc1_meter.avg} "
        f"Acc@5: {acc5_meter.avg}"
    )

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
) -> List[Dict]:
    """
    Generates predictions for the test set, optionally using TTA.

    Args:
        model: The PyTorch model (eval mode).
        data_loader: Test data loader.
        device: Computation device.

    Returns:
        List of dictionaries containing 'image_id' and 'label'.
    """
    model.eval()

    predictions = []

    for batch in data_loader:
        # Handle cases where dataset returns (image, label, image_id) or just (image, label)
        if len(batch) == 3:
            images, _, image_ids = batch
        else:
            images, _ = batch
            image_ids = None  # Should not happen given dataset implementation for test

        images = images.to(device, non_blocking=True)

        # Standard forward pass
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        # Test Time Augmentation (TTA) - Horizontal Flip
        if Config.TTA_FLIP:
            images_flipped = torch.flip(
                images, dims=[3]
            )  # Flip width dimension (B, C, H, W)
            logits_flipped = model(images_flipped)
            probs_flipped = torch.softmax(logits_flipped, dim=1)

            # Average probabilities
            probs = (probs + probs_flipped) / 2.0

        # Get predicted labels
        pred_labels = torch.argmax(probs, dim=1).cpu().numpy()

        if image_ids is not None:
            for img_id, pred in zip(image_ids, pred_labels):
                predictions.append({"image_id": img_id, "label": int(pred)})

    return predictions
