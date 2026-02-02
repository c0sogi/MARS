import time
import torch
import torch.nn as nn
from library.config import CFG
from library.utils import AverageMeter, accuracy
from library.data import Mixup


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    device,
    scheduler=None,
    model_ema=None,
    logger=None,
):
    """
    Trains the model for one epoch using MixUp/CutMix and Gradient Accumulation.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (str): Device to train on.
        scheduler (optional): Learning rate scheduler.
        model_ema (ModelEMA, optional): EMA model wrapper.
        logger (logging.Logger, optional): Logger for output.

    Returns:
        float: Average loss for the epoch.
        float: Average top-1 accuracy for the epoch.
    """
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    # Instantiate Mixup/CutMix handler
    mixup_fn = Mixup(
        mixup_p=CFG.mixup_p,
        cutmix_p=CFG.cutmix_p,
        mixup_alpha=CFG.mixup_alpha,
        cutmix_alpha=CFG.cutmix_alpha,
        num_classes=CFG.num_classes,
    )

    # Soft Target Cross Entropy Loss
    # PyTorch CrossEntropyLoss accepts soft targets (probabilities) of shape (B, C)
    criterion = nn.CrossEntropyLoss()

    # Gradient Accumulation setup
    # train_loader.batch_size is the physical batch size (e.g., 16 or 32)
    # CFG.target_batch_size is the effective batch size we want to achieve (e.g., 32)
    current_batch_size = train_loader.batch_size
    accumulation_steps = max(1, CFG.target_batch_size // current_batch_size)

    optimizer.zero_grad()

    start_time = time.time()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply MixUp/CutMix
        # This returns mixed images and soft (one-hot/mixed) labels
        images, targets = mixup_fn(images, labels)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Perform optimization step
        # We step if we've accumulated enough gradients OR if it's the last batch in the epoch
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

            # Update Model EMA
            if model_ema is not None:
                model_ema.update(model)

        # Metric tracking
        with torch.no_grad():
            # Undo scaling for logging purposes
            batch_loss = loss.item() * accumulation_steps
            losses.update(batch_loss, images.size(0))

            # For accuracy, we compare predictions against the dominant class in the soft targets
            # targets is (B, num_classes)
            hard_targets = targets.argmax(dim=1)
            acc1 = accuracy(outputs, hard_targets, topk=(1,))
            top1.update(acc1[0].item(), images.size(0))

        if logger and (step + 1) % 100 == 0:
            lr_str = ""
            if scheduler:
                try:
                    # Attempt to get LR for logging
                    lr = scheduler.get_last_lr()[0]
                    lr_str = f" LR: {lr:.6f}"
                except:
                    pass

            logger.info(
                f"Epoch: [{epoch}][{step+1}/{len(train_loader)}]"
                f" Loss: {losses.val:.4f} ({losses.avg:.4f})"
                f" Acc@1: {top1.val:.4f} ({top1.avg:.4f}){lr_str}"
            )

    elapsed = time.time() - start_time
    if logger:
        logger.info(
            f"Epoch {epoch} Train finished. Time: {elapsed:.2f}s Loss: {losses.avg:.4f} Acc@1: {top1.avg:.4f}"
        )

    return losses.avg, top1.avg


def validate(model, val_loader, device, logger=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate (can be EMA model).
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to evaluate on.
        logger (logging.Logger, optional): Logger for output.

    Returns:
        float: Average loss.
        float: Average top-1 accuracy.
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    # Standard Cross Entropy for validation (expects hard integer labels)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            acc1 = accuracy(outputs, labels, topk=(1,))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

    elapsed = time.time() - start_time

    if logger:
        # Print full precision as requested
        logger.info(
            f"Validation finished. Time: {elapsed:.2f}s Loss: {losses.avg} Acc@1: {top1.avg}"
        )

    return losses.avg, top1.avg
