import time
import torch
import torch.nn as nn
from library.utils import AverageMeter, accuracy
from library.config import CFG


def train_one_epoch(
    epoch,
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    mixup_fn=None,
    scheduler=None,
):
    """
    Performs one epoch of training.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to train on.
        mixup_fn (Callable, optional): Mixup/CutMix function.
        scheduler (Scheduler, optional): Learning rate scheduler.

    Returns:
        float: Average loss.
        float: Average accuracy.
    """
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply MixUp / CutMix if provided
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        batch_size = images.size(0)

        # Forward pass
        y_preds = model(images)

        # Compute loss
        loss = criterion(y_preds, labels)

        # Scale loss for gradient accumulation
        loss = loss / CFG.accum_iter
        loss.backward()

        # Gradient Accumulation Step
        if (step + 1) % CFG.accum_iter == 0 or (step + 1) == len(dataloader):
            # Gradient Clipping
            if CFG.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # Note: Scheduler stepping is assumed to be handled per-epoch in the main loop
            # or passed appropriately if per-step is required.

        # Update metrics
        # Restore loss scale for logging
        losses.update(loss.item() * CFG.accum_iter, batch_size)

        # Calculate accuracy
        # If MixUp is used, labels are soft [B, C]. Convert to indices for accuracy calc.
        if mixup_fn is not None:
            _, targets_hard = labels.max(dim=1)
        else:
            targets_hard = labels

        acc1 = accuracy(y_preds, targets_hard, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        # Logging
        if step % CFG.print_freq == 0 or (step + 1) == len(dataloader):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(dataloader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc: {top1.val:.4f} ({top1.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return losses.avg, top1.avg


def valid_one_epoch(epoch, model, dataloader, criterion, device):
    """
    Performs one epoch of validation.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation dataloader.
        criterion (Loss): The loss function.
        device (str): Device to validate on.

    Returns:
        float: Average loss.
        float: Average accuracy.
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            # Forward pass
            y_preds = model(images)

            # Compute loss
            loss = criterion(y_preds, labels)

            # Update metrics
            losses.update(loss.item(), batch_size)

            acc1 = accuracy(y_preds, labels, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            # Logging
            if step % CFG.print_freq == 0 or (step + 1) == len(dataloader):
                print(
                    f"EVAL: [{epoch + 1}][{step}/{len(dataloader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc: {top1.val:.4f} ({top1.avg:.4f})"
                )

    return losses.avg, top1.avg
