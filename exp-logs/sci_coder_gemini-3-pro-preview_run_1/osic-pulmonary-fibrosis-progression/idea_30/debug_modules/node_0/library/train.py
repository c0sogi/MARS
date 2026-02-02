import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import CVERNet, laplace_log_likelihood_loss, predict_and_submit
from library.utils import score_function


class LaplaceLikelihoodLoss(nn.Module):
    """
    Wrapper class for the Laplace Log Likelihood loss function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, fvc_true, fvc_pred, sigma, device):
        return laplace_log_likelihood_loss(fvc_true, fvc_pred, sigma, device)


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move inputs to device
        img_ax = batch["img_axial"].to(device)
        img_cor = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        weeks = batch["weeks"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        base_week = batch["base_week"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(
            img_ax, img_cor, tabular, weeks, base_fvc, base_week
        )

        # Calculate loss
        loss = loss_fn(target, fvc_pred, sigma_pred, device)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_preds = []
    val_sigmas = []
    val_targets = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, weeks, base_fvc, base_week
            )

            val_preds.extend(fvc_pred.cpu().numpy())
            val_sigmas.extend(sigma_pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    # Compute metric using the official score function
    return score_function(
        np.array(val_targets), np.array(val_preds), np.array(val_sigmas)
    )


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Main execution function for training and submission generation.
    """
    # Reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Model & Optimization
    model = CVERNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = LaplaceLikelihoodLoss()

    # Training Loop Variables
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.10f}"
        )

        # Early Stopping & Model Checkpointing
        if val_score > best_metric:
            best_metric = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Score: {best_metric:.10f}")

    # Generate Submission
    print("Generating submission...")
    predict_and_submit(test_loader)
