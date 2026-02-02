import torch
import torch.nn as nn
import math
import sys
import numpy as np
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from library.utils import accuracy


def train_one_epoch(model, data_loader, optimizer, device, epoch, config):
    """
    Trains the model for one epoch using MixUp/CutMix and Gradient Clipping.

    Args:
        model: PyTorch model.
        data_loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Calculation device (CPU/GPU).
        epoch: Current epoch number.
        config: Configuration object.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    # Instantiate Mixup/CutMix if enabled in config
    mixup_fn = None
    if config.mixup_p > 0:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_p,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes,
        )

    # Select Loss Function
    # SoftTargetCrossEntropy is required when targets are mixed (probabilities)
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        # Standard CrossEntropy with optional label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    total_loss = 0.0
    num_batches = len(data_loader)

    optimizer.zero_grad()

    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)

        # Apply MixUp/CutMix
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward Pass
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # Gradient Accumulation
        loss = loss / config.accumulate_grad_batches
        loss.backward()

        if (batch_idx + 1) % config.accumulate_grad_batches == 0:
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            optimizer.zero_grad()

        # Record Loss (scale back up for reporting)
        total_loss += loss.item() * config.accumulate_grad_batches

    return total_loss / num_batches


def valid_one_epoch(model, data_loader, device, config):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        data_loader: Validation DataLoader.
        device: Calculation device.
        config: Configuration object.

    Returns:
        tuple: (Average Loss, Average Top-1 Accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = targets.to(device)

            outputs = model(samples)
            loss = criterion(outputs, targets)

            # Calculate Top-1 Accuracy
            # accuracy() returns a list of values for each topk; we take the first (top-1)
            acc = accuracy(outputs, targets, topk=(1,))[0]

            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc
