import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.loss import HierarchicalCompoundLoss
from library.utils import get_weighted_log_loss


def get_optimizer_and_scheduler(model, num_epochs):
    """
    Configures the AdamW optimizer and CosineAnnealingLR scheduler.

    Args:
        model (nn.Module): The model to optimize.
        num_epochs (int): Total number of training epochs.

    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Decoupled Cosine Annealing: T_max set to 1.5x epochs to avoid
    # hitting the absolute minimum LR too early.
    t_max = int(num_epochs * Config.T_MAX_MULT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to train on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Initialize Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Loss function
    criterion = HierarchicalCompoundLoss()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            # Model outputs logits for C1-C7 (Batch, 7)
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, weighted_log_loss_metric)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = HierarchicalCompoundLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass (Logits for C1-C7)
            # Use autocast for consistency, though not strictly necessary for eval
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # --- Prepare Predictions for Metric ---
            # 1. Convert logits to probabilities
            probs_vertebrae = torch.sigmoid(outputs)  # (B, 7)

            # 2. Derive patient_overall probability
            # Logic: max(C1..C7)
            probs_patient = torch.max(probs_vertebrae, dim=1).values.unsqueeze(
                1
            )  # (B, 1)

            # 3. Concatenate to form (B, 8)
            # Order: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
            batch_preds = torch.cat([probs_vertebrae, probs_patient], dim=1)

            all_preds.append(batch_preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Compute Competition Metric
        metric_score = get_weighted_log_loss(y_pred, y_true)
    else:
        metric_score = 0.0

    return avg_loss, metric_score
