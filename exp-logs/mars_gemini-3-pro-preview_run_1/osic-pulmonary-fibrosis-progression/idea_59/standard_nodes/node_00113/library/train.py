import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import seed_everything, get_device, AverageMeter, compute_metric
from library.dataset import get_dataloaders
from library.model import BCSLNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the negative Modified Laplace Log Likelihood as a loss function.
    Minimizing this loss is equivalent to maximizing the competition metric.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, sigma):
        # Clip sigma to minimum 70 ml
        sigma_clipped = torch.clamp(sigma, min=70)

        # Calculate absolute error and clip to 1000 ml
        delta = torch.abs(y_true - y_pred)
        delta = torch.clamp(delta, max=1000)

        # Calculate terms
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_true.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # Metric = -(term1 + term2)
        # Loss = -Metric = term1 + term2
        loss = term1 + term2

        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        delta_week = batch["delta_week"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

        # Calculate Loss
        loss = criterion(target, fvc_pred, sigma_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), axial.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            target = batch["target"].to(device)

            fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

            # compute_metric returns the actual metric value (negative log likelihood)
            metric = compute_metric(target, fvc_pred, sigma_pred)
            metric_meter.update(metric.item(), axial.size(0))

    return metric_meter.avg


def run_training(
    epochs=30, batch_size=16, patience=8, save_path="./working/best_model.pth"
):
    """
    Orchestrates the training process with Early Stopping and Scheduler.
    """
    seed_everything(42)
    device = get_device()

    # Ensure working directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Model
    model = BCSLNet().to(device)

    # Loss
    criterion = LaplaceLogLikelihoodLoss()

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_metric = -float("inf")
    early_stop_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metric = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print("  -> New best model saved!")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Val Metric: {best_metric:.10f}")
    return best_metric
