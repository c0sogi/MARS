import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel
import numpy as np
import pandas as pd

from library.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LR,
    LABEL_SMOOTHING,
)
from library.utils import AverageMeter


def get_optimizer_scheduler(model):
    """
    Initializes the AdamW optimizer and ReduceLROnPlateau scheduler
    based on the configuration parameters.

    Args:
        model (nn.Module): The model to optimize.

    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    return optimizer, scheduler


def bce_with_label_smoothing(logits, targets, smoothing=0.0):
    """
    Computes Binary Cross Entropy loss with label smoothing.

    Args:
        logits (torch.Tensor): Model raw outputs.
        targets (torch.Tensor): Binary targets (0 or 1).
        smoothing (float): Label smoothing factor epsilon.

    Returns:
        torch.Tensor: Scalar loss.
    """
    if smoothing > 0:
        # y_ls = y * (1 - eps) + 0.5 * eps
        targets = targets * (1.0 - smoothing) + 0.5 * smoothing

    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, targets)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Device to run on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter("Train Loss")

    for i, (images, angles, labels, _) in enumerate(dataloader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = bce_with_label_smoothing(outputs, labels, smoothing=LABEL_SMOOTHING)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    # Log epoch summary
    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg:.6f}")

    return loss_meter.avg


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, predictions, targets)
               predictions and targets are numpy arrays.
    """
    model.eval()
    loss_meter = AverageMeter("Val Loss")

    preds = []
    targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, angles, labels, _ in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            # Store probabilities and targets
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    print(f"Validation Loss: {loss_meter.avg:.6f}")

    return loss_meter.avg, np.array(preds), np.array(targets)


def get_swa_model(model):
    """
    Wraps the model in an AveragedModel for Stochastic Weight Averaging.

    Args:
        model (nn.Module): The base model.

    Returns:
        AveragedModel: The SWA model wrapper.
    """
    return AveragedModel(model)


def update_swa_bn(loader, swa_model, device):
    """
    Updates the Batch Normalization statistics for the SWA model.
    This is a custom implementation to handle the specific input signature
    (images, angles) which the default torch.optim.swa_utils.update_bn
    does not support automatically.

    Args:
        loader (DataLoader): DataLoader to compute statistics on.
        swa_model (AveragedModel): The SWA model.
        device (torch.device): Device to run on.
    """
    swa_model.train()

    # Save current momentum to restore later
    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.num_batches_tracked = torch.zeros_like(module.num_batches_tracked)
            momenta[module] = module.momentum
            # Use cumulative moving average
            module.momentum = None

    print("Updating SWA Batch Normalization statistics...")
    with torch.no_grad():
        for images, angles, _, _ in loader:
            images = images.to(device)
            angles = angles.to(device)
            swa_model(images, angles)

    # Restore momentum
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set using TTA (Klein Four-Group)
    and saves the submission file.

    TTA Strategy:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 180 (HFlip + VFlip)

    Args:
        model (nn.Module): The trained model (or SWA model).
        loader (DataLoader): Test data loader.
        device (torch.device): Device to run on.
        output_path (str): Path to save the CSV.
    """
    model.eval()
    ids_all = []
    probs_all = []

    print("Generating submission with TTA...")

    with torch.no_grad():
        for images, angles, _, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # --- TTA 1: Original ---
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # --- TTA 2: Horizontal Flip ---
            # images shape: (B, 3, H, W). W is dim 3.
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # --- TTA 3: Vertical Flip ---
            # H is dim 2.
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # --- TTA 4: Rotate 180 (HFlip + VFlip) ---
            images_hv = torch.flip(images, [2, 3])
            out4 = model(images_hv, angles)
            prob4 = torch.sigmoid(out4)

            # Average predictions
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            probs_all.extend(avg_prob.cpu().numpy().flatten())
            ids_all.extend(ids)

    # Create DataFrame
    df = pd.DataFrame({"id": ids_all, "is_iceberg": probs_all})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
