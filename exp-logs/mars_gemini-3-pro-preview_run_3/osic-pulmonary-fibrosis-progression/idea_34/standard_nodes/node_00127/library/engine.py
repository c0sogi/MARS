import math
import torch
import torch.nn as nn
from library.utils import AverageMeter, score_function
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss for optimization.
    Minimizing this loss is equivalent to maximizing the competition metric.

    Formula: Loss = (sqrt(2) * |y_true - y_pred|) / sigma + log(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, pred_mu, pred_sigma, target):
        """
        Args:
            pred_mu: Predicted FVC (normalized).
            pred_sigma: Predicted uncertainty (normalized, positive).
            target: True FVC (normalized).
        """
        # Constants
        sq2 = math.sqrt(2)

        # Calculate absolute error
        delta = torch.abs(target - pred_mu)

        # Compute NLL terms
        # pred_sigma is guaranteed positive by Softplus in the model
        term1 = (sq2 * delta) / pred_sigma
        term2 = torch.log(sq2 * pred_sigma)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move data to device
        imgs = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_mu, pred_sigma = model(tabular, imgs)

        # Calculate loss
        loss = criterion(pred_mu, pred_sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), imgs.size(0))

    return losses.avg


def evaluate(model, loader, device, scaler_stats):
    """
    Evaluates the model on the validation set using the competition metric.
    Predictions are unnormalized before scoring to match the target scale (ml).
    """
    model.eval()
    scores = AverageMeter()

    # Extract scaler stats for unnormalization
    fvc_mean = scaler_stats["fvc_mean"]
    fvc_std = scaler_stats["fvc_std"]

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            pred_mu, pred_sigma = model(tabular, imgs)

            # Unnormalize predictions and targets to original scale (ml)
            # mu_ml = mu_norm * std + mean
            # sigma_ml = sigma_norm * std
            pred_mu_ml = pred_mu * fvc_std + fvc_mean
            pred_sigma_ml = pred_sigma * fvc_std
            target_ml = targets * fvc_std + fvc_mean

            # Calculate competition metric
            # score_function handles the clipping logic (sigma >= 70, delta <= 1000)
            metric = score_function(target_ml, pred_mu_ml, pred_sigma_ml)

            scores.update(metric, imgs.size(0))

    return scores.avg


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    scaler_stats,
    epochs,
    patience=15,
):
    """
    Manages the full training loop, including validation, scheduler updates,
    early stopping, and saving the best model.
    """
    criterion = LaplaceLogLikelihoodLoss()
    best_score = -float("inf")
    early_stop_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, device, scaler_stats)

        # Step Scheduler (Cosine Annealing is typically updated per epoch)
        if scheduler is not None:
            scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Metric: {val_score}"
        )

        # Checkpoint & Early Stopping
        # Metric is negative and higher is better (closer to 0)
        if val_score > best_score:
            best_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_score
