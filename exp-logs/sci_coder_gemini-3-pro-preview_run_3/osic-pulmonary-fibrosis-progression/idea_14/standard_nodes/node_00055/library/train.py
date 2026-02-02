import os
import torch
import numpy as np
from library.config import Config
from library.model import DDSRNet, get_optimizer_and_scheduler, loss_fn
from library.data import get_dataloaders
from library.utils import AverageMeter, laplace_log_likelihood_metric, seed_everything


def train_epoch(loader, model, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        image = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        mu, sigma = model(image, tabular)

        loss = loss_fn(mu, sigma, target)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), image.size(0))

    return loss_meter.avg


def validate_epoch(loader, model, device):
    """
    Performs validation and computes the competition metric.
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            raw_fvc = batch["raw_fvc"].numpy()  # Ground truth in ml

            mu, sigma = model(image, tabular)

            # Inverse Transform (Un-scale)
            # Convert tensors to numpy
            mu_np = mu.cpu().numpy().flatten()
            sigma_np = sigma.cpu().numpy().flatten()

            # Apply inverse scaling based on Config stats
            mu_unscaled = mu_np * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_unscaled = sigma_np * Config.TARGET_STD

            # Calculate Metric
            score = laplace_log_likelihood_metric(raw_fvc, mu_unscaled, sigma_unscaled)
            metric_meter.update(score, image.size(0))

    return metric_meter.avg


def run_training(patience=15):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DDSRNet().to(device)

    # Get Optimizer and Scheduler (Differential LRs)
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders()

    print(
        f"Starting training on {device} for {Config.EPOCHS} epochs with Early Stopping (Patience={patience})..."
    )

    best_metric = -float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(train_loader, model, optimizer, device)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_metric = validate_epoch(val_loader, model, device)

        # Logging
        print(
            f"Epoch {epoch+1:02d}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")
    return best_metric
