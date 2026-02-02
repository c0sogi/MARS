import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from library.config import Config
from library.utils import calculate_metric


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloader: The training dataloader.
        device: The device to train on.
        epoch: The current epoch number.
        class_weights: Optional tensor of class weights for the loss function.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()

    # Initialize loss function with class weights if provided
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if class_weights is not None:
        criterion = criterion.to(device)

    scaler = torch.cuda.amp.GradScaler()
    running_loss = 0.0
    dataset_size = 0

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        # Note: Scheduler step is typically handled epoch-wise in the main loop
        # for CosineAnnealing, so we do not step it here.

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    return epoch_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        device: The device to evaluate on.
        class_weights: Optional tensor of class weights for the loss function.

    Returns:
        Tuple containing:
            - Average validation loss
            - Average ROC AUC score
            - Numpy array of predicted probabilities
            - Numpy array of ground truth labels
    """
    model.eval()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if class_weights is not None:
        criterion = criterion.to(device)

    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            batch_size = images.size(0)

            y_preds = model(images)
            loss = criterion(y_preds, labels)

            # Apply softmax to get probabilities
            probs = torch.softmax(y_preds, dim=1)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Calculate metric (Mean Column-wise ROC AUC)
    avg_score = calculate_metric(targets, preds)

    return avg_loss, avg_score, preds, targets


def inference_with_tta(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> np.ndarray:
    """
    Performs inference on the test set using Test-Time Augmentation (TTA).
    Strategy: Average predictions of the original image and a horizontal flip.

    Args:
        model: The trained PyTorch model.
        dataloader: The test dataloader (returns only images).
        device: The device to perform inference on.

    Returns:
        np.ndarray: Predicted probabilities of shape (N_samples, N_classes).
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images in dataloader:
            # images shape: (B, C, H, W)
            images = images.to(device, dtype=torch.float)

            # 1. Prediction on Original Image
            out_orig = model(images)
            probs_orig = torch.softmax(out_orig, dim=1)

            # 2. Prediction on Horizontally Flipped Image
            # Flip along the width dimension (dim=3 for BCHW)
            images_flipped = torch.flip(images, dims=[3])
            out_flip = model(images_flipped)
            probs_flip = torch.softmax(out_flip, dim=1)

            # Average the probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            preds_list.append(probs_avg.cpu().numpy())

    final_preds = np.concatenate(preds_list, axis=0)
    return final_preds
