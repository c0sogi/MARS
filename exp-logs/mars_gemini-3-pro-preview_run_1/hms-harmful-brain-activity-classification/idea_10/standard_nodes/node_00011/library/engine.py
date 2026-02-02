import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import AverageMeter, KL_loss


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler (expected OneCycleLR).
        device: The computing device (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # PyTorch's KLDivLoss expects input as Log-Probabilities and Target as Probabilities
    criterion = nn.KLDivLoss(reduction="batchmean")

    for batch_idx, (eeg, spec, targets) in enumerate(loader):
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(eeg, spec)

        # Apply LogSoftmax for numerical stability with KLDivLoss
        log_probs = F.log_softmax(logits, dim=1)

        loss = criterion(log_probs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step scheduler per batch (OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), eeg.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        device: The computing device.

    Returns:
        tuple: (average_loss, average_kl_metric)
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    criterion = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        for eeg, spec, targets in loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(eeg, spec)

            # Loss calculation (LogSoftmax vs Target Probs)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

            # Metric calculation (Probs vs Target Probs)
            # Using the provided utility which expects probabilities
            probs = F.softmax(logits, dim=1)
            kl_score = KL_loss(probs, targets)

            loss_meter.update(loss.item(), eeg.size(0))
            metric_meter.update(kl_score, eeg.size(0))

    return loss_meter.avg, metric_meter.avg


def fit(model, train_loader, val_loader, optimizer, scheduler, device, config=Config):
    """
    Main training loop with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Scheduler.
        device: Device.
        config: Configuration class.
    """
    best_metric = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.get_output_dir(), "best_model.pth")

    print(f"Starting training on device: {device}")

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        # Print metrics with full precision
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val KL Metric: {val_metric}")
        print(f"Learning Rate: {current_lr}")

        # Early Stopping Check
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            print(f"New best model found! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation KL Metric: {best_metric}")
