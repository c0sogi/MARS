import torch
import numpy as np
import sys
from library.utils import AverageMeter, save_model, unscale_target


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The HybridCRNN model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function (e.g., L1Loss).
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (spec, stats, target, _) in enumerate(dataloader):
        # Move data to device
        spec = spec.to(device, non_blocking=True)
        stats = stats.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model expects (spec, stats)
        predictions = model(spec, stats)

        # Compute loss
        loss = criterion(predictions, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), spec.size(0))

    return loss_meter.avg


def evaluate(model, dataloader, criterion, device, target_mean, target_std):
    """
    Evaluates the model on the validation set.
    Computes loss on scaled targets and MAE on unscaled (original) targets.

    Args:
        model (nn.Module): The HybridCRNN model.
        dataloader (DataLoader): Validation dataloader.
        criterion (Loss): Loss function.
        device (str): 'cuda' or 'cpu'.
        target_mean (float): Mean used for scaling targets.
        target_std (float): Std used for scaling targets.

    Returns:
        tuple: (average_scaled_loss, average_unscaled_mae)
    """
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    with torch.no_grad():
        for i, (spec, stats, target, _) in enumerate(dataloader):
            # Move data to device
            spec = spec.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass
            predictions = model(spec, stats)

            # Compute Scaled Loss (for consistency with training loss)
            loss = criterion(predictions, target)
            loss_meter.update(loss.item(), spec.size(0))

            # Compute Unscaled MAE (Competition Metric)
            # Unscale predictions and targets
            # Move to CPU for numpy operations if needed, or keep tensor
            pred_unscaled = unscale_target(predictions, target_mean, target_std)
            target_unscaled = unscale_target(target, target_mean, target_std)

            # Calculate absolute error
            abs_error = torch.abs(pred_unscaled - target_unscaled)
            mae_meter.update(abs_error.mean().item(), spec.size(0))

    return loss_meter.avg, mae_meter.avg


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path,
    target_mean,
    target_std,
    scheduler=None,
):
    """
    Runs the full training loop with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device string.
        epochs: Max epochs.
        patience: Early stopping patience.
        save_path: Path to save best model.
        target_mean: Target scaler mean.
        target_std: Target scaler std.
        scheduler: Learning rate scheduler (optional).
    """
    best_val_mae = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs.")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device, target_mean, target_std
        )

        # Scheduler Step
        if scheduler is not None:
            # Assume ReduceLROnPlateau which needs a metric, or StepLR which doesn't
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mae)
            else:
                scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}: Train Loss (Scaled) = {train_loss}, Val Loss (Scaled) = {val_loss}, Val MAE (Unscaled) = {val_mae}"
        )

        # Early Stopping & Checkpointing
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            save_model(model, save_path)
            print(f"New best model saved with Val MAE: {best_val_mae}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MAE: {best_val_mae}")
