import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter
from library.dataset import MixupCutMix


def train_one_epoch(model, optimizer, dataloader, device, epoch, model_ema=None):
    """
    Trains the model for one epoch using Mixup/CutMix and Label Smoothing.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        dataloader: The training dataloader.
        device: The device to run training on.
        epoch: Current epoch number.
        model_ema: Optional ModelEMA instance to update.

    Returns:
        float: Average training loss.
    """
    model.train()

    loss_meter = AverageMeter()

    # Initialize Mixup/CutMix augmentation
    mixup_fn = MixupCutMix(
        mixup_alpha=Config.mixup_alpha,
        cutmix_alpha=Config.cutmix_alpha,
        prob=Config.mixup_prob,
        num_classes=Config.num_classes,
    )

    # Define Loss Function
    # We use BCEWithLogitsLoss with positive weights to handle class imbalance
    # pos_weight must be a vector of length num_classes
    pos_weight = torch.full((Config.num_classes,), Config.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup or CutMix
        images, targets = mixup_fn(images, targets)

        # Apply Label Smoothing
        # Since targets might be mixed (float), we apply smoothing linearly:
        # new_target = target * (1 - epsilon) + 0.5 * epsilon
        if Config.label_smoothing > 0:
            targets = (
                targets * (1 - Config.label_smoothing) + 0.5 * Config.label_smoothing
            )

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Update Model EMA
        if model_ema is not None:
            model_ema.update(model)

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model: The model to validate (can be the EMA model).
        dataloader: The validation dataloader.
        device: The device to run validation on.

    Returns:
        tuple: (average_loss, predictions, ground_truth)
    """
    model.eval()

    loss_meter = AverageMeter()

    # Define Loss Function for validation
    # We keep pos_weight for consistent loss scaling, but skip label smoothing
    # to measure the error against the true labels.
    pos_weight = torch.full((Config.num_classes,), Config.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    preds = []
    valid_labels = []

    with torch.no_grad():
        for step, (images, targets, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            preds.append(probs.cpu().numpy())
            valid_labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    print(f"Validation Loss: {loss_meter.avg}")

    return loss_meter.avg, preds, valid_labels
