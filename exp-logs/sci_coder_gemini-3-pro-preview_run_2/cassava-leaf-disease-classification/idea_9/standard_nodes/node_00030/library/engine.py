import time
import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from library.utils import AverageMeter, accuracy
from library.config import CFG


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    device,
    scheduler=None,
    mixup_fn=None,
    model_ema=None,
):
    """
    Performs one epoch of training.
    Handles MixUp/CutMix, Dynamic Loss selection, Gradient Accumulation, and EMA updates.
    """
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    # Instantiate loss functions
    # SoftTargetCrossEntropy is used when Mixup/Cutmix is active (targets are probabilities)
    criterion_soft = SoftTargetCrossEntropy()
    # LabelSmoothingCrossEntropy is used when standard training is active (targets are integers)
    criterion_smooth = LabelSmoothingCrossEntropy(smoothing=CFG.label_smoothing)

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply MixUp / CutMix if function is provided
        if mixup_fn is not None:
            images, mixed_targets = mixup_fn(images, targets)
            logits = model(images)
            loss = criterion_soft(logits, mixed_targets)
            # For accuracy monitoring, compare against the dominant class in the mixed target
            acc_targets = mixed_targets.argmax(dim=1)
        else:
            logits = model(images)
            loss = criterion_smooth(logits, targets)
            acc_targets = targets

        # Track metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        acc1 = accuracy(logits, acc_targets, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        # specific handling for gradient accumulation
        loss = loss / CFG.grad_accum_steps
        loss.backward()

        # Step optimizer and update EMA
        if (batch_idx + 1) % CFG.grad_accum_steps == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

    print(f"Epoch {epoch}: Train Loss {losses.avg:.4f} | Train Acc {top1.avg:.4f}")

    return losses.avg, top1.avg


def valid_one_epoch(epoch, model, val_loader, device):
    """
    Performs validation on the hold-out set.
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    # Use Label Smoothing loss for validation to match training objective
    criterion = LabelSmoothingCrossEntropy(smoothing=CFG.label_smoothing)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = images.size(0)
            acc1 = accuracy(logits, targets, topk=(1,))

            losses.update(loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

    # Print full precision as requested
    print(f"Epoch {epoch}: Valid Loss {losses.avg} | Valid Acc {top1.avg}")

    return losses.avg, top1.avg
