import torch
import torch.nn as nn
import numpy as np
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from library.config import Config
from library.utils import AverageMeter


def get_mixup_fn():
    """
    Factory function to create the Mixup/CutMix callable based on Config.

    Returns:
        Mixup: Configured Mixup instance from timm.
    """
    return Mixup(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=Config.LABEL_SMOOTHING,
        num_classes=Config.NUM_CLASSES,
    )


def train_one_epoch(
    epoch, model, loader, optimizer, device, loss_fn, scaler, mixup_fn=None
):
    """
    Executes one training epoch with AMP and MixUp/CutMix.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to compute on.
        loss_fn (callable): Loss function (usually SoftTargetCrossEntropy).
        scaler (GradScaler): AMP GradScaler for mixed precision training.
        mixup_fn (callable, optional): Mixup function to apply to batch.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply MixUp / CutMix if configured
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()

        # Automatic Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), inputs.size(0))

    # Print metrics with full precision
    print(f"Train Epoch: {epoch} Loss: {loss_meter.avg}")

    return loss_meter.avg


def valid_one_epoch(epoch, model, loader, device, loss_fn):
    """
    Executes one validation epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        device (torch.device): Device to compute on.
        loss_fn (callable): Loss function (usually CrossEntropyLoss).

    Returns:
        tuple: (avg_loss, avg_acc, predictions, targets)
            - avg_loss (float): Average validation loss.
            - avg_acc (float): Average validation accuracy.
            - predictions (np.ndarray): Raw softmax probabilities (N, num_classes).
            - targets (np.ndarray): True labels (N,).
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))

            # Calculate Accuracy
            # Note: During validation, targets are class indices (not mixed)
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            acc_meter.update(acc.item(), inputs.size(0))

            # Store Softmax Probabilities for OOF/Stacking
            probs = torch.softmax(outputs, dim=1)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    # Aggregate results
    predictions = np.concatenate(preds_list)
    valid_targets = np.concatenate(targets_list)

    # Print metrics with full precision
    print(f"Valid Epoch: {epoch} Loss: {loss_meter.avg} Accuracy: {acc_meter.avg}")

    return loss_meter.avg, acc_meter.avg, predictions, valid_targets
