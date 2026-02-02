import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV2
from timm.loss import SoftTargetCrossEntropy
from library.config import CFG
from library.utils import AverageMeter, accuracy


def train_one_epoch(
    epoch,
    model,
    optimizer,
    data_loader,
    device,
    model_ema=None,
    mixup_fn=None,
    scaler=None,
):
    """
    Trains the model for one epoch.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Define loss function for soft targets (used when MixUp is active)
    soft_ce = SoftTargetCrossEntropy()

    for step, (images, labels) in enumerate(data_loader):
        data_time.update(time.time() - start)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply MixUp/CutMix if function is provided
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # Forward pass with Automatic Mixed Precision (AMP)
        # Using torch.amp.autocast for compatibility with modern PyTorch versions
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            output = model(images)

            # Compute loss based on label type
            if labels.ndim == 2:
                # Soft labels (from MixUp/CutMix)
                loss = soft_ce(output, labels)
            else:
                # Hard integer labels
                loss = F.cross_entropy(output, labels)

            # Normalize loss for gradient accumulation
            loss = loss / CFG.gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (handling gradient accumulation)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            # Update Model EMA
            if model_ema is not None:
                model_ema.update(model)

        # Update metrics
        # Multiply loss back by accumulation steps to log the actual batch loss
        losses.update(loss.item() * CFG.gradient_accumulation_steps, images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if step % CFG.print_freq == 0 or step == (len(data_loader) - 1):
            print(
                f"Epoch: [{epoch}][{step}/{len(data_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def valid_one_epoch(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(images)
            loss = F.cross_entropy(output, labels)

            acc1 = accuracy(output, labels, topk=(1,))[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))

    # Print full precision metrics
    print(f"Validation: Loss {losses.avg} Acc@1 {top1.avg}")

    return losses.avg, top1.avg


def predict(model, data_loader, device):
    """
    Generates predictions for the test set.
    Applies Test Time Augmentation (TTA) if configured.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device, non_blocking=True)

            # 1. Original Prediction
            output = model(images)
            output = F.softmax(output, dim=1)

            # 2. TTA: Horizontal Flip
            if CFG.tta_steps > 1:
                # Flip images horizontally (dim 3 is width)
                images_flip = torch.flip(images, dims=[3])
                output_flip = model(images_flip)
                output_flip = F.softmax(output_flip, dim=1)

                # Average predictions
                output = (output + output_flip) / 2.0

            preds.append(output.cpu())

    return torch.cat(preds, dim=0)
