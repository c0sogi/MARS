import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import AverageMeter, calculate_accuracy
from library.config import Config


def get_optimizer_llrd(model: nn.Module, config: Config, learning_rate: float):
    """
    Constructs the AdamW optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model (nn.Module): The model to optimize.
        config (Config): Configuration object.
        learning_rate (float): The base learning rate.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    layer_decay = config.layer_decay
    weight_decay = config.weight_decay

    # Define parameter groups with specific learning rates
    # ConvNeXt structure: backbone.stem, backbone.stages.0..3, backbone.norm, head
    # We assign 'depth' relative to the head (depth 0).

    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine LLRD scale based on parameter name
        if "head" in name:
            depth = 0
        elif "backbone.norm" in name:
            depth = 1
        elif "backbone.stages.3" in name:
            depth = 2
        elif "backbone.stages.2" in name:
            depth = 3
        elif "backbone.stages.1" in name:
            depth = 4
        elif "backbone.stages.0" in name:
            depth = 5
        elif "backbone.stem" in name:
            depth = 6
        else:
            # Fallback for any other parameters (e.g. if model structure differs)
            depth = 6

        scale = layer_decay**depth

        # Determine Weight Decay
        # Apply no weight decay to biases and 1D parameters (like LayerNorm weights)
        if param.ndim <= 1 or "bias" in name:
            p_wd = 0.0
        else:
            p_wd = weight_decay

        param_groups.append(
            {"params": [param], "lr": learning_rate * scale, "weight_decay": p_wd}
        )

    optimizer = torch.optim.AdamW(
        param_groups, lr=learning_rate, eps=config.opt_eps, betas=config.opt_betas
    )

    return optimizer


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    scaler: torch.amp.GradScaler,
    model_ema=None,
    mixup_fn=None,
    grad_accum_steps: int = 1,
):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to compute on.
        config (Config): Configuration object.
        scaler (GradScaler): Mixed precision scaler.
        model_ema (ModelEMA, optional): EMA model wrapper.
        mixup_fn (Mixup, optional): Mixup/CutMix function.
        grad_accum_steps (int): Steps for gradient accumulation.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    losses = AverageMeter()

    optimizer.zero_grad()

    num_steps = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup/CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        else:
            # If no mixup, convert indices to one-hot for SoftTargetCrossEntropy
            if len(targets.shape) == 1:
                targets = F.one_hot(targets, num_classes=config.num_classes).float()

        # Mixed Precision Forward
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Normalize loss for gradient accumulation
        loss = loss / grad_accum_steps

        # Backward
        scaler.scale(loss).backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_steps:
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

            # Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA
            if model_ema is not None:
                model_ema.update(model)

        # Update metrics (scale loss back up for logging)
        losses.update(loss.item() * grad_accum_steps, inputs.size(0))

        if batch_idx % config.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{num_steps}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device.
        config (Config): Configuration object.

    Returns:
        float: Average Loss.
        float: Accuracy.
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Convert targets to one-hot for SoftTargetCrossEntropy
            targets_one_hot = F.one_hot(targets, num_classes=config.num_classes).float()

            # Forward
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets_one_hot)

            # Measure accuracy using raw indices
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc, inputs.size(0))

    print(f" * Validation Results - Loss: {losses.avg:.6f} | Accuracy: {top1.avg:.6f}")
    return losses.avg, top1.avg
