import time
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import AverageMeter, accuracy
from library.loss import SoftTargetCrossEntropy


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
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


def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, accum_steps=1
):
    """
    Trains the model for one epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    criterion = SoftTargetCrossEntropy()

    num_steps = len(dataloader)
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Prepare targets (One-Hot Encoding)
        labels_one_hot = F.one_hot(labels, num_classes=Config.num_classes).float()

        # MixUp / CutMix Logic
        use_mixup_cutmix = Config.enable_mixup_cutmix
        current_targets = None

        if use_mixup_cutmix:
            # Decide between MixUp and CutMix based on mixup_prob
            if np.random.rand() < Config.mixup_prob:
                # MixUp
                lam = np.random.beta(Config.mixup_alpha, Config.mixup_alpha)
                index = torch.randperm(batch_size).to(device)

                mixed_images = lam * images + (1 - lam) * images[index, :]
                mixed_targets = (
                    lam * labels_one_hot + (1 - lam) * labels_one_hot[index, :]
                )
            else:
                # CutMix
                lam = np.random.beta(Config.cutmix_alpha, Config.cutmix_alpha)
                index = torch.randperm(batch_size).to(device)

                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

                # Adjust lambda to match exact pixel ratio
                lam = 1 - (
                    (bbx2 - bbx1)
                    * (bby2 - bby1)
                    / (images.size()[-1] * images.size()[-2])
                )

                mixed_images = images.clone()
                mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[
                    index, :, bbx1:bbx2, bby1:bby2
                ]
                mixed_targets = (
                    lam * labels_one_hot + (1 - lam) * labels_one_hot[index, :]
                )

            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_targets)
            current_targets = mixed_targets
        else:
            outputs = model(images)
            loss = criterion(outputs, labels_one_hot)
            current_targets = labels_one_hot

        # Normalize loss for gradient accumulation
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update metrics
        with torch.no_grad():
            loss_meter.update(loss.item() * accum_steps, batch_size)
            # Calculate accuracy (using argmax of soft targets if mixed)
            acc = accuracy(outputs, current_targets)[0]
            acc_meter.update(acc.item(), batch_size)

        if (step + 1) % Config.print_freq == 0:
            print(
                f"Epoch: [{epoch + 1}][{step + 1}/{num_steps}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"Acc: {acc_meter.avg:.2f}"
            )

    return loss_meter.avg, acc_meter.avg


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    criterion = SoftTargetCrossEntropy()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            labels_one_hot = F.one_hot(labels, num_classes=Config.num_classes).float()

            outputs = model(images)
            loss = criterion(outputs, labels_one_hot)

            loss_meter.update(loss.item(), batch_size)
            acc = accuracy(outputs, labels_one_hot)[0]
            acc_meter.update(acc.item(), batch_size)

    # Print full precision as requested
    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation Acc: {acc_meter.avg}")

    return loss_meter.avg, acc_meter.avg


def update_bn(loader, model, device):
    """
    Updates Batch Normalization running statistics by doing a forward pass.
    """
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            model(images)


def run_swa(model, dataloader, checkpoints, device):
    """
    Averages model weights from checkpoints and updates BN statistics.

    Args:
        model: Base model instance to load weights into.
        dataloader: DataLoader for updating BN statistics (usually train loader).
        checkpoints: List of file paths to model checkpoints.
        device: Torch device.
    """
    if not checkpoints:
        print("No checkpoints provided for SWA.")
        return model

    print(f"SWA: Averaging {len(checkpoints)} checkpoints...")

    avg_state_dict = None

    for path in checkpoints:
        print(f"Loading {path}...")
        checkpoint = torch.load(path, map_location="cpu")

        # Extract state_dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if avg_state_dict is None:
            avg_state_dict = copy.deepcopy(state_dict)
        else:
            for k, v in state_dict.items():
                # Only accumulate floating point tensors (weights/biases)
                # Skip integer buffers (like num_batches_tracked)
                if v.is_floating_point():
                    avg_state_dict[k] += v

    # Average the weights
    for k in avg_state_dict.keys():
        if avg_state_dict[k].is_floating_point():
            avg_state_dict[k] /= len(checkpoints)

    model.load_state_dict(avg_state_dict)
    model.to(device)

    print("SWA: Updating BN statistics...")
    update_bn(dataloader, model, device)

    return model
