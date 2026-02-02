import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from library.utils import AverageMeter
from library.config import CFG


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.

    Args:
        size (torch.Size): Size of the image tensor (B, C, H, W).
        lam (float): Lambda value from Beta distribution.

    Returns:
        tuple: Bounding box coordinates (bbx1, bby1, bbx2, bby2).
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup and CutMix augmentations.
    """
    model.train()
    losses = AverageMeter()

    # Check if augmentation is enabled
    apply_aug = CFG.mixup_prob > 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Determine if we apply Mixup or CutMix
        do_mixup = False
        do_cutmix = False

        if apply_aug and np.random.rand() < CFG.mixup_prob:
            if np.random.rand() < 0.5:
                do_mixup = True
            else:
                do_cutmix = True

        if do_mixup:
            # Mixup
            alpha = CFG.mixup_alpha
            # Sample lambda from Beta distribution
            lam = np.random.beta(alpha, alpha)

            # Shuffle indices
            index = torch.randperm(batch_size).to(device)

            # Mix images
            mixed_images = lam * images + (1 - lam) * images[index, :]

            # Mix targets
            targets_a, targets_b = labels, labels[index]

            # Forward pass
            outputs = model(mixed_images)
            outputs = outputs.squeeze(1)

            # Calculate mixed loss
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )

        elif do_cutmix:
            # CutMix
            alpha = CFG.cutmix_alpha
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(batch_size).to(device)

            # Generate bounding box
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

            # Apply CutMix
            images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

            # Adjust lambda to match exact pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
            )

            targets_a, targets_b = labels, labels[index]

            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(1)

            # Calculate mixed loss
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )

        else:
            # Standard Training
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)

        losses.update(loss.item(), batch_size)

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def valid_one_epoch(epoch, model, loader, criterion, device):
    """
    Validates the model for one epoch and calculates Log Loss.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            losses.update(loss.item(), batch_size)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate Log Loss
    # Clip predictions to avoid log(0) errors, though sklearn usually handles this
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
    score = log_loss(targets, preds_clipped)

    print(f"Validation Epoch {epoch}: Loss = {losses.avg}, Log Loss = {score}")

    return losses.avg, preds


def inference_fn(model, loader, device):
    """
    Performs inference on the test set, optionally using Test-Time Augmentation (TTA).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for step, (images, ids) in enumerate(loader):
            images = images.to(device)

            # Forward pass (Original)
            outputs = model(images)
            outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs)

            if CFG.tta:
                # Horizontal Flip TTA
                # Flip along width dimension (dim 3 for NCHW)
                images_flipped = torch.flip(images, dims=[3])

                outputs_flipped = model(images_flipped)
                outputs_flipped = outputs_flipped.squeeze(1)
                probs_flipped = torch.sigmoid(outputs_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds)
    return preds
