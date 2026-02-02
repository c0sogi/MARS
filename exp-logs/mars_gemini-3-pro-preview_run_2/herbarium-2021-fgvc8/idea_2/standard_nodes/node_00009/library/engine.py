import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Tuple, Dict, Any

from library.config import Config
from library.utils import AverageMeter, calculate_f1


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP) and CutMix.

    Args:
        model: The neural network model.
        loader: DataLoader yielding (images, targets_dict) via CutMixCollator.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        criterion: Loss function (HierarchicalLoss).
        device: Device string ('cuda' or 'cpu').
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    losses = AverageMeter()
    scaler = GradScaler()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        # Move targets to device
        # targets is a dict: {'species': (t_a, t_b), 'family': (t_a, t_b), ...}
        for key, val in targets.items():
            if isinstance(val, tuple) or isinstance(val, list):
                # val is (tensor_a, tensor_b)
                targets[key] = (
                    val[0].to(device, non_blocking=True),
                    val[1].to(device, non_blocking=True),
                )
            elif isinstance(val, torch.Tensor):
                targets[key] = val.to(device, non_blocking=True)
            # 'lam' is a float/scalar, usually doesn't need explicit .to() if used as python float,
            # but if it's in the dict it might be just a float.
            # The Loss class expects it in the dict.

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            # outputs = (species_logits, family_logits, order_logits)
            outputs = model(images)

            # Compute Loss
            loss = criterion(outputs, targets)

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Step optimizer
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader yielding (image, species, family, order).
        criterion: Loss function.
        device: Device string.

    Returns:
        Tuple[float, float]: (Macro F1 Score, Average Loss)
    """
    model.eval()

    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species, families, orders in loader:
            images = images.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            families = families.to(device, non_blocking=True)
            orders = orders.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            species_logits, family_logits, order_logits = outputs

            # Calculate validation loss
            # Construct targets dict format expected by HierarchicalLoss
            # Since no CutMix in validation, target_a == target_b and lam=1.0
            targets = {
                "species": (species, species),
                "family": (families, families),
                "order": (orders, orders),
                "lam": 1.0,
            }

            loss = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))

            # Get predictions (argmax of species logits)
            preds = torch.argmax(species_logits, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(species.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate Metric
    f1 = calculate_f1(all_targets, all_preds)

    return f1, losses.avg
