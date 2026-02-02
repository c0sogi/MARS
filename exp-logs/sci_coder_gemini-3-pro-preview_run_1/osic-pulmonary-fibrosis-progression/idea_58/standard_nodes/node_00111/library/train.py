import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, AverageMeter, Logger, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import BBSLNet

# ==========================================
# Loss Function
# ==========================================


class LaplaceLoss(nn.Module):
    """
    Differentiable Loss function mirroring the competition metric.
    Minimizes the negative Log Likelihood with clipping constraints.
    """

    def __init__(self):
        super().__init__()
        self.sigma_min = Config.SIGMA_MIN
        self.error_max = Config.ERROR_MAX
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, y_true, y_pred, sigma):
        """
        Args:
            y_true: Ground truth FVC
            y_pred: Predicted FVC
            sigma: Predicted Confidence
        """
        device = y_true.device
        self.sqrt_2 = self.sqrt_2.to(device)

        # Apply constraints
        sigma_clipped = torch.clamp(sigma, min=self.sigma_min)
        abs_diff = torch.abs(y_true - y_pred)
        delta = torch.clamp(abs_diff, max=self.error_max)

        # Calculate NLL terms
        # Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # Loss = - Metric
        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2
        return torch.mean(loss)


# ==========================================
# Training & Validation Steps
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move inputs to device
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)

        # Metadata for parametric reconstruction
        # meta['Base_FVC'] and meta['Weeks'] are tensors in the batch
        base_fvc = batch["meta"]["Base_FVC"].to(device).float()
        weeks = batch["meta"]["Weeks"].to(device).float()

        optimizer.zero_grad()

        # Forward Pass: Predict Parameters
        # alpha: slope, sigma_base: base uncertainty, sigma_growth: uncertainty growth
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Parametric Reconstruction
        # FVC = Base + alpha * delta_t
        fvc_pred = base_fvc + alpha * weeks

        # Confidence = Base_Sigma + Growth_Sigma * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

        # Calculate Loss
        loss = criterion(target, fvc_pred, sigma_pred)

        # Backward Pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def evaluate(model, loader, device):
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            base_fvc = batch["meta"]["Base_FVC"].to(device).float()
            weeks = batch["meta"]["Weeks"].to(device).float()

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Reconstruction
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Calculate Metric (Higher is better)
            score = laplace_log_likelihood(target, fvc_pred, sigma_pred)
            metric_meter.update(score.item(), img_ax.size(0))

    return metric_meter.avg


# ==========================================
# Main Training Routine
# ==========================================


def train_model(debug=False):
    seed_everything(Config.SEED)

    # Logging
    log_file = os.path.join(Config.WORKING_DIR, "train_log.txt")
    logger = Logger(log_file)
    logger.log(f"Starting training for Idea: {Config.IDEA_ID}")
    logger.log(f"Device: {Config.DEVICE}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Model
    model = BBSLNet().to(Config.DEVICE)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    criterion = LaplaceLoss()

    # Tracking
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        logger.log(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )

        # Validate
        val_metric = evaluate(model, val_loader, Config.DEVICE)

        # Update Scheduler
        scheduler.step()

        logger.log(f"Train Loss: {train_loss:.6f}")
        logger.log(f"Val Metric: {val_metric}")  # Full precision

        # Checkpointing & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"New Best Model Saved! Score: {best_metric}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.log(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.log("Early stopping triggered.")
            break

    logger.log(f"Training Complete. Best Validation Metric: {best_metric}")
    return best_model_path


# ==========================================
# Inference & Submission
# ==========================================


def generate_submission(model_path, debug=False):
    print("\nGenerating Submission...")

    # Load Data
    _, _, test_loader = get_dataloaders(debug=debug)

    # Load Model
    model = BBSLNet().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(Config.DEVICE)
            img_cor = batch["image_coronal"].to(Config.DEVICE)
            tabular = batch["tabular"].to(Config.DEVICE)

            # Metadata
            base_fvc = batch["meta"]["Base_FVC"].to(Config.DEVICE).float()
            weeks = batch["meta"]["Weeks"].to(Config.DEVICE).float()
            patient_weeks = batch["meta"]["Patient_Week"]  # List of strings

            # Predict Parameters
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Reconstruct
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Clip Confidence strictly for submission
            sigma_pred = torch.clamp(sigma_pred, min=Config.SIGMA_MIN)

            # Collect results
            fvc_np = fvc_pred.cpu().numpy()
            sigma_np = sigma_pred.cpu().numpy()

            for i in range(len(patient_weeks)):
                results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": fvc_np[i],
                        "Confidence": sigma_np[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure columns are correct
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(sub_df.head())


# ==========================================
# Entry Point Logic
# ==========================================


def run(debug=False):
    # Train
    best_model_path = train_model(debug=debug)

    # Predict
    generate_submission(best_model_path, debug=debug)
