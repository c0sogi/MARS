import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter
from library.loss import DiceBCELoss


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (DataLoader): Training data loader.
        device (torch.device): Compute device (CPU or GPU).
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    criterion = DiceBCELoss()
    scaler = torch.cuda.amp.GradScaler()

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision (AMP) Context
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def valid_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Computes the Global Dice Coefficient.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        tuple: (Average Validation Loss, Global Dice Score)
    """
    model.eval()

    loss_meter = AverageMeter()
    criterion = DiceBCELoss()

    # Accumulators for Global Dice Calculation
    # Metric: 2 * |X n Y| / (|X| + |Y|) over the entire dataset
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            # Test Time Augmentation (TTA)
            if Config.USE_TTA:
                # 1. Original
                logits = model(images)

                # 2. Horizontal Flip
                images_hflip = torch.flip(images, dims=[3])
                logits_hflip = model(images_hflip)
                logits_hflip = torch.flip(logits_hflip, dims=[3])

                # 3. Vertical Flip
                images_vflip = torch.flip(images, dims=[2])
                logits_vflip = model(images_vflip)
                logits_vflip = torch.flip(logits_vflip, dims=[2])

                # Average predictions
                logits = (logits + logits_hflip + logits_vflip) / 3.0
            else:
                logits = model(images)

            # Compute Loss
            loss = criterion(logits, masks)
            loss_meter.update(loss.item(), images.size(0))

            # Compute Global Dice Statistics
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            # Binarize predictions
            preds = (probs > 0.5).float()

            # Flatten tensors for set operations
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

    # Calculate Global Dice
    # Add smooth to denominator to avoid division by zero if both sets are empty
    global_dice = (2.0 * total_intersection) / (total_union + Config.SMOOTH)

    return loss_meter.avg, global_dice
