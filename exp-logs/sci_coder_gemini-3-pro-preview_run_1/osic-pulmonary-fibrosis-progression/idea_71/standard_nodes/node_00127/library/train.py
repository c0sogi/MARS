import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import DBSLNet
from library.utils import metric_laplace_log_likelihood


class CustomLoss(nn.Module):
    """
    Differentiable implementation of the modified Laplace Log Likelihood metric.
    Loss = -Metric, so minimizing this maximizes the score.

    Formula:
        Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
        Loss   = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)

    Where:
        delta = min(|true - pred|, 1000)
        sigma_clipped = max(sigma, 70)
    """

    def __init__(self):
        super().__init__()
        self.max_error = float(Config.MAX_ERROR)
        self.min_sigma = float(Config.MIN_CONFIDENCE)
        # Register sqrt(2) as a buffer so it moves with the model
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, fvc_pred, sigma_pred, fvc_true):
        # Calculate absolute error
        abs_error = torch.abs(fvc_true - fvc_pred)

        # Apply error thresholding (delta)
        delta = torch.clamp(abs_error, max=self.max_error)

        # Apply confidence clipping
        sigma_clipped = torch.clamp(sigma_pred, min=self.min_sigma)

        # Compute terms
        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Sum terms to get negative metric (Loss)
        loss = term1 + term2

        # Return mean loss over batch
        return torch.mean(loss)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        week = batch["week"].to(device)
        base_week = batch["base_week"].to(device)
        base_fvc = batch["base_fvc"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(
            img_ax, img_cor, tabular, week, base_week, base_fvc
        )

        # Compute loss
        loss = criterion(fvc_pred, sigma_pred, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        running_loss += loss.item() * img_ax.size(0)

    # Return average loss
    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, device):
    """
    Performs validation using the official metric function.
    """
    model.eval()
    total_metric = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Move data to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            base_week = batch["base_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, week, base_week, base_fvc
            )

            # Calculate metric using utility function
            # Note: utility expects tensors or numpy arrays and returns mean float
            score = metric_laplace_log_likelihood(target, fvc_pred, sigma_pred)

            # Accumulate weighted score
            batch_size = img_ax.size(0)
            total_metric += score * batch_size
            num_samples += batch_size

    # Return average metric
    return total_metric / num_samples


def run_training(debug=False, epochs=None):
    """
    Main training pipeline.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
        epochs (int): Number of epochs to train. Defaults to Config.EPOCHS.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    if epochs is None:
        epochs = Config.EPOCHS

    # 2. Data Loading
    train_loader, val_loader = get_dataloaders(debug=debug)

    # 3. Model Initialization
    model = DBSLNet()
    model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = CustomLoss()
    criterion.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop with Early Stopping
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print(f"Training for {epochs} epochs with patience {Config.PATIENCE}...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Timing
        elapsed = time.time() - start_time

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}s - Train Loss: {train_loss} - Val Score: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            print(
                f"Validation Score improved from {best_score} to {val_score}. Saving model to {best_model_path}..."
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
