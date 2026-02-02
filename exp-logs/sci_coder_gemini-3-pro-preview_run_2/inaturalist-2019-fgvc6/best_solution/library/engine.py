import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import CFG
from library.utils import AverageMeter, accuracy


def mixup_data(x, y, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (DataLoader): The training data loader.
        device (torch.device): The device to run training on.
        epoch (int): The current epoch number.

    Returns:
        tuple: (average_loss, average_top1_accuracy)
    """
    model.train()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # Use label smoothing as per configuration for training
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
    scaler = GradScaler()

    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training with Mixup
        with autocast():
            if np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, device, alpha=1.0
                )
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                # Approximate accuracy for logging
                acc1_a, acc5_a = accuracy(outputs, targets_a, topk=(1, 5))
                acc1_b, acc5_b = accuracy(outputs, targets_b, topk=(1, 5))
                acc1 = lam * acc1_a + (1 - lam) * acc1_b
                acc5 = lam * acc5_a + (1 - lam) * acc5_b
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1.item(), batch_size)
        top5_meter.update(acc5.item(), batch_size)

    print(
        f"Epoch: [{epoch}] Train Loss: {loss_meter.avg} Train Top1: {top1_meter.avg} Train Top5: {top5_meter.avg}"
    )
    return loss_meter.avg, top1_meter.avg


def validate(model, data_loader, device, tta=False):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.
        tta (bool): Whether to use Test Time Augmentation (Horizontal Flip).

    Returns:
        tuple: (average_loss, average_top1_accuracy)
    """
    model.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # Standard CrossEntropyLoss for validation to reflect true error
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Use autocast for inference efficiency
            with autocast():
                outputs = model(images)

                if tta:
                    # Horizontal Flip TTA
                    outputs += model(torch.flip(images, dims=[3]))
                    outputs /= 2.0

                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)

            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(acc1.item(), batch_size)
            top5_meter.update(acc5.item(), batch_size)

    print(
        f"Validation Loss: {loss_meter.avg} Validation Top1: {top1_meter.avg} Validation Top5: {top5_meter.avg}"
    )
    return loss_meter.avg, top1_meter.avg
