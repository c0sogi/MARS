import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from library.utils import AverageMeter, accuracy
from library.config import Config


def train_one_epoch(
    epoch,
    model,
    optimizer,
    data_loader,
    device,
    scheduler=None,
    mixup_fn=None,
    model_ema=None,
    accum_iter=1,
    label_smoothing=0.0,
):
    """
    Trains the model for one epoch.
    Handles dynamic loss selection, gradient accumulation, and EMA updates.
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # Select criterion based on whether Mixup/Cutmix/LabelSmoothing is active.
    # If mixup_fn is present, targets are transformed to (Batch, Num_Classes) with soft labels.
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer.zero_grad()
    num_steps = len(data_loader)

    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)

        # Apply Mixup/Cutmix if function is provided
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        # Normalize loss for gradient accumulation
        loss_scaled = loss / accum_iter
        loss_scaled.backward()

        # Step optimizer and scheduler only after accumulation steps
        if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == num_steps:
            optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step()

        # Update tracking metrics
        # Scale loss back up for logging
        loss_meter.update(loss.item(), samples.size(0))

        # Calculate accuracy
        # If targets are mixed/smoothed (dim > 1), use argmax for approximation
        if targets.dim() > 1:
            acc_targets = targets.argmax(dim=1)
        else:
            acc_targets = targets

        acc_res = accuracy(outputs, acc_targets, topk=(1,))
        acc_meter.update(acc_res[0].item(), samples.size(0))

    # Print full precision metrics as requested
    print(f"Epoch {epoch} Training: Loss {loss_meter.avg} Accuracy {acc_meter.avg}")
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    for batch in data_loader:
        # Ensure we can handle the batch structure
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            samples, targets = batch[0], batch[1]
        else:
            continue

        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        acc_res = accuracy(outputs, targets, topk=(1,))

        loss_meter.update(loss.item(), samples.size(0))
        acc_meter.update(acc_res[0].item(), samples.size(0))

    print(f"Validation: Loss {loss_meter.avg} Accuracy {acc_meter.avg}")
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def inference(model, data_loader, device, tta=False):
    """
    Performs inference on the data_loader.
    Supports Test Time Augmentation (Horizontal Flip).
    """
    model.eval()
    final_preds = []

    for batch in data_loader:
        # Handle cases where dataloader returns (images, labels) or just images
        if isinstance(batch, (tuple, list)):
            samples = batch[0]
        else:
            samples = batch

        samples = samples.to(device)

        # Original forward pass
        outputs = model(samples)
        probs = torch.softmax(outputs, dim=1)

        if tta:
            # Horizontal flip TTA
            # dim 3 is width (N, C, H, W)
            samples_flipped = torch.flip(samples, dims=[3])
            outputs_flipped = model(samples_flipped)
            probs_flipped = torch.softmax(outputs_flipped, dim=1)

            # Average probabilities
            probs = (probs + probs_flipped) / 2.0

        final_preds.append(probs.cpu())

    return torch.cat(final_preds, dim=0)
