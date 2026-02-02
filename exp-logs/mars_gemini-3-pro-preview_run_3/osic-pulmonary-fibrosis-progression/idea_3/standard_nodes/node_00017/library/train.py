import os
import numpy as np
import torch
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, calculate_metric, seed_everything
from library.data import get_dataloaders
from library.model import MultiViewNet


def laplace_nll_loss(mu, sigma, target):
    """
    Calculates the Negative Log Likelihood (NLL) for a Laplace distribution.

    The metric is defined based on a modified Laplace likelihood.
    Maximizing the metric is equivalent to minimizing this loss.

    Loss = (sqrt(2) * |target - mu|) / sigma + log(sigma)

    Args:
        mu (torch.Tensor): Predicted mean (standardized).
        sigma (torch.Tensor): Predicted standard deviation (standardized).
        target (torch.Tensor): Ground truth value (standardized).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Calculate absolute error
    delta = torch.abs(target - mu)

    # Calculate NLL terms
    # We ignore the constant term log(sqrt(2)) as it doesn't affect gradients
    loss = (np.sqrt(2) * delta) / sigma + torch.log(sigma)

    return torch.mean(loss)


def train_one_epoch(train_loader, model, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, tabular, targets) in enumerate(train_loader):
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # mu and sigma are standardized
        mu, sigma = model(images, tabular)

        # Calculate loss on standardized values
        loss = laplace_nll_loss(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(val_loader, model, device):
    """
    Evaluates the model on the validation set.
    Performs inverse transformation to calculate the metric in original units (ml).
    """
    model.eval()
    metric_meter = AverageMeter()

    # Constants for inverse transformation
    target_mean = Config.target_mean
    target_std = Config.target_std

    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            mu_std, sigma_std = model(images, tabular)

            # --- Inverse Transformation ---
            # Convert standardized predictions back to ml
            # mu_real = mu_std * std + mean
            mu_real = mu_std * target_std + target_mean

            # sigma_real = sigma_std * std (Scale only)
            sigma_real = sigma_std * target_std

            # Convert standardized targets back to ml for metric calculation
            targets_real = targets * target_std + target_mean

            # Calculate metric (higher is better)
            # calculate_metric handles the clipping (max(sigma, 70)) internally
            score = calculate_metric(targets_real, mu_real, sigma_real)

            metric_meter.update(score, images.size(0))

    return metric_meter.avg


def run_training():
    """
    Main training function.
    Initializes model, data, and optimizer.
    Runs the training loop with early stopping and checkpointing.
    """
    # Reproducibility
    seed_everything(Config.seed)

    # Setup hardware
    device = torch.device(Config.device)

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers
    )

    # Model
    model = MultiViewNet()
    model = model.to(device)

    # Optimizer
    # Using Adam as a standard robust optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    # Scheduler (Cite Lesson 11 - Convergence Calibration)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=1e-5
    )

    # Training State
    best_metric = -float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"Starting training on {device} for {Config.epochs} epochs...")
    print(f"Model: {Config.backbone_name} (Frozen Backbone: {Config.freeze_backbone})")

    for epoch in range(1, Config.epochs + 1):
        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, device, epoch)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_metric = validate(val_loader, model, device)

        # Logging
        print(
            f"Epoch [{epoch}/{Config.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_path = os.path.join(Config.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best metric! Model saved to {save_path}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Training complete. Best Metric: {best_metric:.10f} at Epoch {best_epoch}")
    return best_metric
