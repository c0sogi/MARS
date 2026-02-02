import torch
import numpy as np
import os
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import CG_SDAN, criterion


class CustomLoss(torch.nn.Module):
    """
    Wrapper for the differentiable Modified Laplace Log Likelihood Loss.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        alpha,
        sigma_base,
        sigma_growth,
        target_fvc,
        meta_weeks,
        meta_base_fvc,
        meta_base_week,
    ):
        return criterion(
            alpha,
            sigma_base,
            sigma_growth,
            target_fvc,
            meta_weeks,
            meta_base_fvc,
            meta_base_week,
        )


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)

        # Extract metadata for trajectory calculation
        m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
        m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
        m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

        # Calculate loss
        loss = loss_fn(
            alpha, s_base, s_growth, targets, m_weeks, m_base_fvc, m_base_week
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    true_fvc = []
    pred_fvc = []
    pred_sigma = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
            m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
            m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)

            # Forward pass
            alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

            # Reconstruct predictions from parametric outputs
            # FVC = Base + alpha * delta_t
            delta_t = m_weeks - m_base_week
            p_fvc = m_base_fvc + alpha * delta_t

            # Sigma = Base + Growth * |delta_t|
            p_sigma = s_base + s_growth * torch.abs(delta_t)

            # Collect results
            true_fvc.extend(targets.cpu().numpy().flatten())
            pred_fvc.extend(p_fvc.cpu().numpy().flatten())
            pred_sigma.extend(p_sigma.cpu().numpy().flatten())

    # Compute metric using the gathered arrays
    metric = compute_metric(
        np.array(true_fvc), np.array(pred_fvc), np.array(pred_sigma)
    )
    return metric


def run_training(debug=False):
    """
    Orchestrates the training process including setup, loop, and early stopping.
    """
    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Initialize DataLoaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Initialize Model
    model = CG_SDAN().to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss Function
    loss_fn = CustomLoss()

    # Training State
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.N_EPOCHS} epochs...")

    for epoch in range(Config.N_EPOCHS):
        # Train
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn
        )

        # Validate
        val_metric = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Log metrics (full precision)
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val Metric: {val_metric}"
        )

        # Early Stopping and Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Metric: {best_metric}")
