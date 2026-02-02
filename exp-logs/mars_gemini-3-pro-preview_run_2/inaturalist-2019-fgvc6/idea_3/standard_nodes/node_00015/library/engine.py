import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from library.config import CFG
from library.utils import AverageMeter, accuracy


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

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute metrics
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1.item(), batch_size)
        top5_meter.update(acc5.item(), batch_size)

    print(
        f"Epoch: [{epoch}] Train Loss: {loss_meter.avg} Train Top1: {top1_meter.avg} Train Top5: {top5_meter.avg}"
    )
    return loss_meter.avg, top1_meter.avg


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.

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
