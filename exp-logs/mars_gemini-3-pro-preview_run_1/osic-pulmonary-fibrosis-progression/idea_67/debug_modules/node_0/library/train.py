import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import AverageMeter, score_function
from library.model import TSCGNet, get_extended_dataloaders


class LaplaceLoss(nn.Module):
    """
    Calculates the negative modified Laplace Log Likelihood loss.

    The loss is derived from the metric:
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    We minimize Loss = -Metric.
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()
        self.confidence_clip = Config.CONFIDENCE_CLIP
        self.max_error = Config.MAX_ERROR

    def forward(
        self, alpha, sigma_base, sigma_growth, delta_week, baseline_fvc, target
    ):
        """
        Args:
            alpha: Predicted slope (B,)
            sigma_base: Predicted base confidence (B,)
            sigma_growth: Predicted confidence growth rate (B,)
            delta_week: Time difference from baseline (B,)
            baseline_fvc: Baseline FVC measurement (B,)
            target: True FVC measurement (B,)
        """
        # 1. Calculate predictions based on anchored trajectory logic
        fvc_pred = baseline_fvc + alpha * delta_week
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        # 2. Clip confidence values
        sigma_clipped = torch.clamp(sigma_pred, min=self.confidence_clip)

        # 3. Calculate absolute error and clip it
        delta = torch.abs(target - fvc_pred)
        delta = torch.clamp(delta, max=self.max_error)

        # 4. Calculate Metric
        # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
        term1 = (np.sqrt(2) * delta) / sigma_clipped
        term2 = torch.log(np.sqrt(2) * sigma_clipped)

        # Metric is negative, so Loss = -Metric = term1 + term2
        loss = term1 + term2

        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move data to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        delta_week = batch["delta_week"].to(device)
        target = batch["target"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass: Get parameters alpha, sigma_base, sigma_growth
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Calculate loss
        loss = criterion(
            alpha, sigma_base, sigma_growth, delta_week, baseline_fvc, target
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            target = batch["target"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate predictions for metric
            fvc_pred = baseline_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Calculate score
            score = score_function(target, fvc_pred, sigma_pred)
            scores.update(score, img_ax.size(0))

    return scores.avg


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    patience=Config.PATIENCE,
):
    """
    Main training loop with Early Stopping.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        debug (bool): If True, uses a subset of data.
        patience (int): Early stopping patience.

    Returns:
        str: Path to the best saved model.
    """
    # Set debug flag in Config to ensure dataloaders respect it
    Config.DEBUG = debug

    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")
    print(
        f"Configuration: Epochs={epochs}, Batch Size={batch_size}, Debug={debug}, Patience={patience}"
    )

    # 1. Data Loaders
    train_loader, val_loader, _ = get_extended_dataloaders(batch_size=batch_size)

    # 2. Model setup
    model = TSCGNet().to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLoss()

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_model_path
