import numpy as np
import torch
import torch.nn as nn
from library.utils import AverageMeter, get_score


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config,
) -> float:
    """
    Trains the model for one epoch using the Hierarchical Multi-Task loss.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer (e.g., AdamW).
        scheduler: Learning rate scheduler (stepped batch-wise).
        device: Computation device (CPU/GPU).
        config: Configuration object containing hyperparameters.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    # Define Loss Functions
    # Primary Task: Binary Classification (Malignant vs Benign)
    # Using pos_weight to handle significant class imbalance (~1:55)
    pos_weight = torch.tensor([config.pos_weight], device=device)
    criterion_primary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Auxiliary Task: Multi-class Classification (Diagnosis)
    criterion_aux = nn.CrossEntropyLoss()

    for batch in loader:
        # Move data to device
        images = batch["image"].to(device)
        meta = batch["meta"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Shape: (B, 1)
        diagnoses = batch["diagnosis"].to(device)  # Shape: (B,)

        optimizer.zero_grad()

        # Forward pass
        # Returns: primary_logits (Malignancy), aux_logits (Diagnosis)
        primary_logits, aux_logits = model(images, meta)

        # Compute Losses
        loss_primary = criterion_primary(primary_logits, targets)
        loss_aux = criterion_aux(aux_logits, diagnoses)

        # Combined Loss: Primary + Weighted Auxiliary
        loss = loss_primary + (config.aux_loss_weight * loss_aux)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Optimization Step
        optimizer.step()

        # Scheduler Step (Batch-wise for Cosine Annealing with Warmup)
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, config
) -> tuple:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: Computation device.
        config: Configuration object.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    losses = AverageMeter()

    # Define Loss Functions (mirrored from training for consistent reporting)
    pos_weight = torch.tensor([config.pos_weight], device=device)
    criterion_primary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_aux = nn.CrossEntropyLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            diagnoses = batch["diagnosis"].to(device)

            # Forward pass
            primary_logits, aux_logits = model(images, meta)

            # Compute Loss
            loss_primary = criterion_primary(primary_logits, targets)
            loss_aux = criterion_aux(aux_logits, diagnoses)
            loss = loss_primary + (config.aux_loss_weight * loss_aux)

            losses.update(loss.item(), images.size(0))

            # Predictions for AUC
            # Apply sigmoid to convert logits to probabilities [0, 1]
            preds = torch.sigmoid(primary_logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Aggregate predictions
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    # Flatten arrays to ensure correct shape for roc_auc_score
    auc = get_score(all_targets.flatten(), all_preds.flatten())

    # Print validation metric with full precision as requested
    print(f"Validation AUC: {auc}")

    return losses.avg, auc
