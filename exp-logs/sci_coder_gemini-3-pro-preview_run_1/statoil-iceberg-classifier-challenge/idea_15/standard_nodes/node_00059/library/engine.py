import torch
import torch.nn as nn
import numpy as np
import sys
from library.config import Config
from library.utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        # Model expects (x, angle)
        logits = model(images, angles)

        # Ensure labels are (B, 1) to match logits
        targets = labels.view(-1, 1)

        # Manual Label Smoothing (Cite debug_lesson_5)
        if Config.LABEL_SMOOTHING > 0:
            targets = (
                targets * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )

        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    # Print full precision metric
    print(f"Epoch: {epoch} Train Loss: {losses.avg}")
    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, all logits, and all targets.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, angles)
            loss = criterion(logits, labels.view(-1, 1))

            losses.update(loss.item(), images.size(0))

            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)

    print(f"Val Loss: {losses.avg}")
    return losses.avg, all_logits, all_targets


def run_swa_step(swa_model, model):
    """
    Updates the SWA model parameters with the current model's parameters.
    """
    swa_model.update_parameters(model)


def update_bn(loader, model, device):
    """
    Custom Batch Normalization update for models with dual inputs (image, angle).
    Adapts torch.optim.swa_utils.update_bn logic.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            momenta[module] = module.momentum
            if module.momentum is not None:
                module.momentum = None

    model.train()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)

            # Forward pass to update running stats
            model(images, angles)

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = momenta[module]


def predict_tta(model, loader, device):
    """
    Performs Test Time Augmentation (TTA) prediction.
    Augmentations: Original, Horizontal Flip, Vertical Flip.
    Returns averaged logits and IDs.
    """
    model.eval()
    all_logits = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            ids = batch["id"]

            # 1. Original
            out1 = model(images, angles)

            # 2. Horizontal Flip (Flip width dimension, index 3: B, C, H, W)
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)

            # 3. Vertical Flip (Flip height dimension, index 2: B, C, H, W)
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)

            # Average logits
            avg_logits = (out1 + out2 + out3) / 3.0

            all_logits.append(avg_logits.cpu().numpy())
            all_ids.extend(ids)

    all_logits = np.concatenate(all_logits)
    all_ids = np.array(all_ids)

    return all_logits, all_ids
