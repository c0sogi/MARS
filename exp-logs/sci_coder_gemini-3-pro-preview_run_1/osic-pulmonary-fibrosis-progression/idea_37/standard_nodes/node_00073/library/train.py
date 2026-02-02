import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import seed_everything, AverageMeter, score_function
from library.data import get_dataloaders
from library.model import DALANet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Loss function corresponding to the negative of the competition metric.
    Metric: - (sqrt(2) * Delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    Loss: (sqrt(2) * Delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, sigma):
        # y_pred, y_true, sigma are tensors

        # Sigma clipping: max(sigma, 70)
        sigma_clipped = torch.clamp(sigma, min=70.0)

        # Absolute error
        diff = torch.abs(y_true - y_pred)

        # Error clipping: min(|true - pred|, 1000)
        delta = torch.clamp(diff, max=1000.0)

        # Calculate Loss terms
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device).view(-1, 1)

        # Extract meta features needed for forward pass
        # meta is a dict of lists/tuples, need to convert to tensor
        delta_week = (
            torch.tensor(batch["meta"]["Delta_Week"], dtype=torch.float32)
            .to(device)
            .view(-1, 1)
        )
        baseline_fvc = (
            torch.tensor(batch["meta"]["Baseline_FVC"], dtype=torch.float32)
            .to(device)
            .view(-1, 1)
        )

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, baseline_fvc)

        # Compute loss
        loss = criterion(fvc_pred, target, sigma_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), axial.size(0))

    return losses.avg


def valid_one_epoch(model, loader, device):
    """
    Performs validation and calculates the competition metric.
    """
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device).view(-1, 1)

            delta_week = (
                torch.tensor(batch["meta"]["Delta_Week"], dtype=torch.float32)
                .to(device)
                .view(-1, 1)
            )
            baseline_fvc = (
                torch.tensor(batch["meta"]["Baseline_FVC"], dtype=torch.float32)
                .to(device)
                .view(-1, 1)
            )

            # Forward pass
            fvc_pred, sigma_pred = model(
                axial, coronal, tabular, delta_week, baseline_fvc
            )

            # Compute metric using utility function
            # score_function expects numpy or tensors and handles the math
            metric_val = score_function(target, fvc_pred, sigma_pred)

            scores.update(metric_val, axial.size(0))

    return scores.avg


def run_training():
    """
    Orchestrates the training pipeline with Early Stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")
    Config.print_config()

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(Config)

    # 3. Model
    model = DALANet()
    model.to(device)

    # 4. Optimizer & Scheduler & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = valid_one_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss} | "
            f"Val Score: {val_score}"
        )

        # Early Stopping Logic
        if val_score > best_score:
            print(
                f"Validation Score improved from {best_score} to {val_score}. Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")
    print(f"Best model saved to: {best_model_path}")
