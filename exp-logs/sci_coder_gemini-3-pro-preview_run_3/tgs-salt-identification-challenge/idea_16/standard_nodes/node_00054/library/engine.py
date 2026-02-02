import torch
import torch.cuda.amp as amp
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_map


def center_crop(tensor, target_h, target_w):
    """
    Center crops a tensor to the target height and width.

    Args:
        tensor (torch.Tensor): Input tensor of shape (..., H, W).
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        torch.Tensor: Cropped tensor.
    """
    if tensor.ndim < 2:
        return tensor

    h, w = tensor.shape[-2:]

    if h == target_h and w == target_w:
        return tensor

    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2

    return tensor[..., start_h : start_h + target_h, start_w : start_w + target_w]


def train_one_epoch(
    model, loader, optimizer, scaler, loss_fn, device, epoch, phase_name
):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        loss_fn: Loss function.
        device: Device to train on.
        epoch: Current epoch number.
        phase_name: 'phase1' or 'phase2'.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    # Determine Deep Supervision mode based on phase
    # Phase 1: Deep Supervision Active (returns list of outputs)
    # Phase 2: Deep Supervision Disabled (returns single output)
    deep_supervision = phase_name == "phase1"

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with amp.autocast(enabled=True):
            # Forward pass
            outputs = model(inputs, deep_supervision=deep_supervision)

            # Calculate loss
            # If deep_supervision is True, outputs is a list. DeepSupervisionLoss handles it.
            # If deep_supervision is False, outputs is a tensor. LovaszHingeLoss handles it.
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate_one_epoch(model, loader, loss_fn, device, phase_name):
    """
    Validates the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader.
        loss_fn: Loss function.
        device: Device to validate on.
        phase_name: 'phase1' or 'phase2'.

    Returns:
        tuple: (average_loss, average_map)
    """
    model.eval()
    losses = AverageMeter()
    map_score = AverageMeter()

    # Match training behavior for loss calculation logic
    deep_supervision = phase_name == "phase1"

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs, deep_supervision=deep_supervision)

            # Calculate Loss (consistent with training objective)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            # Prepare for Metric Calculation (mAP)
            # We always evaluate mAP on the final output head
            if isinstance(outputs, (list, tuple)):
                final_output = outputs[-1]
            else:
                final_output = outputs

            # 1. Apply Sigmoid (Logits -> Probabilities)
            probs = torch.sigmoid(final_output)

            # 2. Crop to original resolution (128x128 -> 101x101)
            # This removes the reflection padding added during preprocessing
            probs_cropped = center_crop(
                probs, Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG
            )
            targets_cropped = center_crop(
                targets, Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG
            )

            # 3. Convert to Numpy for calculation
            # Shape: (B, 1, H, W) -> (B, H, W)
            probs_np = probs_cropped.squeeze(1).cpu().numpy()
            targets_np = targets_cropped.squeeze(1).cpu().numpy()

            # 4. Calculate mAP
            batch_map = calculate_map(probs_np, targets_np)
            map_score.update(batch_map, inputs.size(0))

    # Print metrics with full precision
    print(f"Validation Phase: {phase_name} | Loss: {losses.avg} | mAP: {map_score.avg}")

    return losses.avg, map_score.avg
