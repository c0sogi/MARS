import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import SoftTargetCrossEntropy
from library.utils import AverageMeter
from library.config import CFG


def train_one_epoch(
    epoch, model, train_loader, optimizer, device, scheduler=None, accum_iter=1
):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to compute on.
        scheduler (lr_scheduler, optional): Learning rate scheduler.
        accum_iter (int): Gradient accumulation steps.
    """
    model.train()

    # Loss function for soft targets (MixUp/CutMix)
    # If MixUp/CutMix is not applied, the collate fn still returns soft targets (one-hot with smoothing)
    criterion = SoftTargetCrossEntropy()

    losses = AverageMeter()

    # Scaler for AMP
    scaler = torch.amp.GradScaler("cuda")

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / accum_iter

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Record loss (scale back up for logging)
        losses.update(loss.item() * accum_iter, images.size(0))

        if batch_idx % CFG.print_freq == 0 or batch_idx == len(train_loader) - 1:
            # We do not print progress bars, but we can print periodic status if needed.
            # However, instructions say "Only print the required information".
            # Usually, training logs per epoch are sufficient, but debug logs are helpful.
            # We will rely on the main script to print summary, or print here minimally.
            pass

    print(f"Epoch {epoch} Train Loss: {losses.avg}")
    return losses.avg


def valid_one_epoch(epoch, model, val_loader, device):
    """
    Validates the model on the validation set.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to compute on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()

    # Validation targets are standard integers, so we use standard CrossEntropy
    criterion = nn.CrossEntropyLoss()

    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean()

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))

    print(f"Epoch {epoch} Validation Loss: {losses.avg}")
    print(f"Epoch {epoch} Validation Accuracy: {accuracies.avg}")

    return losses.avg, accuracies.avg


def inference_fn(model, test_loader, device, tta_steps=3):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to compute on.
        tta_steps (int): Number of TTA steps (1=Original, 2=Orig+HFlip, 3=Orig+HFlip+VFlip).

    Returns:
        np.ndarray: Predicted labels.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # 1. Original
            out = model(images)
            probs = F.softmax(out, dim=1)

            # 2. Horizontal Flip
            if tta_steps >= 2:
                images_hf = torch.flip(images, dims=[3])
                out_hf = model(images_hf)
                probs += F.softmax(out_hf, dim=1)

            # 3. Vertical Flip
            if tta_steps >= 3:
                images_vf = torch.flip(images, dims=[2])
                out_vf = model(images_vf)
                probs += F.softmax(out_vf, dim=1)

            # Average probabilities
            # Note: If tta_steps=1, we divide by 1. If 3, divide by 3.
            # Actually, for argmax, division doesn't change the max index,
            # but for correctness of probability values we divide.
            probs /= tta_steps

            # Get predicted labels
            batch_preds = probs.argmax(dim=1).cpu().numpy()
            preds.append(batch_preds)

    return np.concatenate(preds)
