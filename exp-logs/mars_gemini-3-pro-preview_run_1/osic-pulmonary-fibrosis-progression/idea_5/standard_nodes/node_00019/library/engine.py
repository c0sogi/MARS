import torch
import os
import numpy as np
from library.utils import AverageMeter, calculate_metric
from library.loss import ModifiedLaplaceLoss


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device to run training on.
        loss_fn: Instance of ModifiedLaplaceLoss.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in dataloader:
        # Move data to device
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        time = batch["time"].to(device)
        target_fvc = batch["fvc"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model outputs: alpha (slope), sigma_base, sigma_growth
        alpha, sigma_base, sigma_growth = model(images, tabular)

        # Calculate Loss
        loss = loss_fn(alpha, sigma_base, sigma_growth, time, baseline_fvc, target_fvc)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update stats
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def evaluate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: Device to run evaluation on.
        loss_fn: Instance of ModifiedLaplaceLoss.

    Returns:
        tuple: (Average Loss, Average Metric)
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            time = batch["time"].to(device)
            target_fvc = batch["fvc"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(images, tabular)

            # Calculate Loss
            loss = loss_fn(
                alpha, sigma_base, sigma_growth, time, baseline_fvc, target_fvc
            )
            loss_meter.update(loss.item(), images.size(0))

            # Calculate Metric
            # Reconstruct predictions based on linear model
            # FVC_pred = FVC_base + alpha * t
            pred_fvc = baseline_fvc + alpha * time

            # Sigma_pred = Sigma_base + Sigma_growth * |t|
            pred_sigma = sigma_base + sigma_growth * torch.abs(time)

            # Compute competition metric
            score = calculate_metric(target_fvc, pred_fvc, pred_sigma)
            metric_meter.update(score, images.size(0))

    return loss_meter.avg, metric_meter.avg


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler (can be None).
        device: Device to run on.
        num_epochs: Maximum number of epochs.
        patience: Number of epochs to wait for improvement before stopping.
        save_path: Path to save the best model weights.
    """
    loss_fn = ModifiedLaplaceLoss()
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validation Step
        val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)

        # Scheduler Step
        if scheduler:
            scheduler.step()

        # Logging
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Metric: {val_metric}")

        # Early Stopping Logic
        # Metric is negative LLL, so higher is better (closer to 0)
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")
