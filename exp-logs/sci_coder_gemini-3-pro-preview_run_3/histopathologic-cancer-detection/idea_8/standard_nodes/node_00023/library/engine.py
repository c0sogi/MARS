import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from library.config import Config
from library.utils import calculate_auc


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: The training data loader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run training on.
        scheduler: Optional learning rate scheduler.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        # Squeeze to match label shape [B] if model outputs [B, 1]
        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        # Step scheduler if it is batch-level (optional, based on usage)
        # Note: Config suggests CosineAnnealing which is typically epoch-level,
        # so we primarily expect the caller to step the scheduler.

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: The validation data loader.
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        Tuple[float, float]: A tuple containing (average_loss, auc_score).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Calculate AUC
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        auc = calculate_auc(all_labels, all_preds)
    else:
        auc = 0.5

    return avg_loss, auc


def predict_tta(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tta_steps: int = 4,
) -> np.ndarray:
    """
    Generates predictions using Test Time Augmentation (TTA).

    Strategies:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 90

    Args:
        model: The PyTorch model.
        dataloader: The data loader for inference.
        device: The device to run inference on.
        tta_steps: Number of TTA views to average (default: 4).

    Returns:
        np.ndarray: The averaged predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            # Accumulator for probabilities
            batch_probs = torch.zeros(batch_size, device=device)

            # View 1: Original
            out = model(images).view(-1)
            batch_probs += torch.sigmoid(out)

            if tta_steps >= 2:
                # View 2: Horizontal Flip (dim 3 is width)
                img_hflip = torch.flip(images, dims=[3])
                out = model(img_hflip).view(-1)
                batch_probs += torch.sigmoid(out)

            if tta_steps >= 3:
                # View 3: Vertical Flip (dim 2 is height)
                img_vflip = torch.flip(images, dims=[2])
                out = model(img_vflip).view(-1)
                batch_probs += torch.sigmoid(out)

            if tta_steps >= 4:
                # View 4: Rotate 90 degrees
                img_rot90 = torch.rot90(images, k=1, dims=[2, 3])
                out = model(img_rot90).view(-1)
                batch_probs += torch.sigmoid(out)

            # Average the probabilities
            batch_probs /= tta_steps
            all_preds.append(batch_probs.cpu().numpy())

    if len(all_preds) > 0:
        return np.concatenate(all_preds)
    else:
        return np.array([])
