import time
import torch
import torch.nn as nn
from library.utils import AverageMeter, accuracy
from library.config import Config


def train_one_epoch(epoch, model, train_loader, optimizer, device, mixup_fn=None):
    """
    Performs one epoch of training.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to run training on.
        mixup_fn (Mixup, optional): Mixup/CutMix function.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    # CrossEntropyLoss supports soft targets (probabilities) which Mixup produces
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup/CutMix if function is provided
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch: {epoch} Train Loss: {losses.avg}")
    return losses.avg


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Top-1 Accuracy.
    """
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            # targets are class indices in validation, so accuracy util works directly
            acc1 = accuracy(output, targets, topk=(1,))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

    # Print full precision as requested
    print(f"Validation Loss: {losses.avg} Acc@1: {top1.avg}")
    return top1.avg


def inference_tta(model, test_loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Applies 4-fold TTA: Original, Horizontal Flip, Vertical Flip, Transpose.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Tensor of predicted probabilities [N, Num_Classes].
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images in test_loader:
            # Handle dataset returning (img, label) or just img
            if isinstance(images, (list, tuple)):
                images = images[0]

            images = images.to(device)

            # 1. Original
            out_1 = model(images)
            prob_1 = torch.softmax(out_1, dim=1)

            # 2. Horizontal Flip (flip width dim)
            out_2 = model(torch.flip(images, dims=[3]))
            prob_2 = torch.softmax(out_2, dim=1)

            # 3. Vertical Flip (flip height dim)
            out_3 = model(torch.flip(images, dims=[2]))
            prob_3 = torch.softmax(out_3, dim=1)

            # 4. Transpose (Swap H and W)
            out_4 = model(torch.transpose(images, 2, 3))
            prob_4 = torch.softmax(out_4, dim=1)

            # Average predictions
            avg_prob = (prob_1 + prob_2 + prob_3 + prob_4) / 4.0
            preds.append(avg_prob.cpu())

    return torch.cat(preds)
