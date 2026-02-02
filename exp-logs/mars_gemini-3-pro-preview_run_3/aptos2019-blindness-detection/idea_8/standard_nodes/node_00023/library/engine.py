import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, quadratic_weighted_kappa


def train_one_epoch(
    model, loader, optimizer, device, epoch, accum_iter=1, scheduler=None
):
    """
    Trains the model for one epoch using Gradient Accumulation.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run training on.
        epoch (int): The current epoch number (for logging).
        accum_iter (int): Number of batches to accumulate gradients before stepping.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    losses = AverageMeter()
    criterion = nn.MSELoss()

    # Initialize gradients
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Reshape outputs to match labels (B, 1) -> (B,)
        outputs = outputs.view(-1)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Normalize loss for gradient accumulation
        loss = loss / accum_iter

        # Backward pass
        loss.backward()

        # Update weights every accum_iter batches
        if (batch_idx + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update metrics (multiply back by accum_iter to log the actual loss per batch)
        losses.update(loss.item() * accum_iter, images.size(0))

    # Handle remaining gradients if the dataset size isn't divisible by accum_iter
    if len(loader) % accum_iter != 0:
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} Train Loss: {losses.avg}")
    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes MSE Loss and Quadratic Weighted Kappa.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: (average_loss, qwk_score)
    """
    model.eval()

    losses = AverageMeter()
    criterion = nn.MSELoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)
            losses.update(loss.item(), images.size(0))

            # Store predictions and labels for QWK calculation
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Post-process predictions for QWK
    # 1. Clip predictions to valid range [0, 4]
    # 2. Round to nearest integer
    final_preds = np.round(np.clip(all_preds, 0, 4)).astype(int)
    final_labels = all_labels.astype(int)

    # Calculate Metric
    qwk_score = quadratic_weighted_kappa(final_labels, final_preds)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation QWK: {qwk_score}")

    return losses.avg, qwk_score
