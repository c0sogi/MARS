import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math

from library.config import Config, seed_everything
from library.utils import score_function
from library.data import get_dataloaders
from library.model import DualAxisNet


class CombinedLoss(nn.Module):
    """
    Computes the weighted sum of the Modified Laplace Log Likelihood (Metric Loss)
    and the Auxiliary MSE Loss for regularization.

    Metric Loss = - ( - (sqrt(2) * Delta) / Sigma_clipped - ln(sqrt(2) * Sigma_clipped) )
                = (sqrt(2) * Delta) / Sigma_clipped + ln(sqrt(2) * Sigma_clipped)

    Where:
        Delta = min(|True_FVC - Pred_FVC|, 1000)
        Sigma_clipped = max(Sigma, 70)
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.aux_weight = Config.AUX_LOSS_WEIGHT
        self.max_error = Config.MAX_ERROR
        self.min_confidence = Config.MIN_CONFIDENCE

    def forward(self, outputs, targets):
        # 1. Unpack Outputs
        # alpha: slope, sigma_base: base uncertainty, sigma_growth: uncertainty growth
        alpha = outputs["alpha"]
        sigma_base = outputs["sigma_base"]
        sigma_growth = outputs["sigma_growth"]
        pred_percent = outputs["pred_percent"]

        # 2. Unpack Targets
        true_fvc = targets["fvc"]
        base_fvc = targets["base_fvc"]
        week_delta = targets["week_delta"]
        true_base_pct = targets["base_pct"]

        # 3. Calculate Parametric Predictions
        # FVC_pred = Base_FVC + alpha * t
        fvc_pred = base_fvc + alpha * week_delta

        # Confidence = sigma_base + sigma_growth * |t|
        confidence = sigma_base + sigma_growth * torch.abs(week_delta)

        # 4. Compute Metric Loss (Negative Log Likelihood)
        # Clip confidence
        sigma_clipped = torch.clamp(confidence, min=self.min_confidence)

        # Calculate absolute error and clip
        abs_error = torch.abs(true_fvc - fvc_pred)
        delta = torch.clamp(abs_error, max=self.max_error)

        # Metric formula terms
        sqrt_2 = math.sqrt(2)
        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # We want to maximize the metric, so we minimize the negative of it.
        # Original Metric = - term1 - term2
        # Loss = - (Original Metric) = term1 + term2
        metric_loss = torch.mean(term1 + term2)

        # 5. Compute Auxiliary Loss (MSE on Baseline Percent)
        aux_loss = self.mse(pred_percent, true_base_pct)

        # 6. Total Loss
        total_loss = metric_loss + self.aux_weight * aux_loss

        return total_loss, metric_loss, aux_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_metric_loss = 0.0
    running_aux_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack inputs
        axial_img, coronal_img, tabular = inputs
        axial_img = axial_img.to(device)
        coronal_img = coronal_img.to(device)
        tabular = tabular.to(device)

        # Move targets to device
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        # Forward pass
        outputs = model(axial_img, coronal_img, tabular)

        # Loss calculation
        loss, m_loss, a_loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * axial_img.size(0)
        running_metric_loss += m_loss.item() * axial_img.size(0)
        running_aux_loss += a_loss.item() * axial_img.size(0)

    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_metric_loss = running_metric_loss / dataset_size
    epoch_aux_loss = running_aux_loss / dataset_size

    return epoch_loss, epoch_metric_loss, epoch_aux_loss


def validate_one_epoch(model, loader, device):
    model.eval()

    all_true_fvc = []
    all_pred_fvc = []
    all_confidence = []

    with torch.no_grad():
        for inputs, targets in loader:
            axial_img, coronal_img, tabular = inputs
            axial_img = axial_img.to(device)
            coronal_img = coronal_img.to(device)
            tabular = tabular.to(device)

            # Targets for calculation
            base_fvc = targets["base_fvc"].to(device)
            week_delta = targets["week_delta"].to(device)
            true_fvc = targets["fvc"].numpy()  # Keep as numpy for accumulation

            # Forward
            outputs = model(axial_img, coronal_img, tabular)

            alpha = outputs["alpha"]
            sigma_base = outputs["sigma_base"]
            sigma_growth = outputs["sigma_growth"]

            # Calculate Predictions
            fvc_pred = base_fvc + alpha * week_delta
            confidence = sigma_base + sigma_growth * torch.abs(week_delta)

            # Store results
            all_true_fvc.extend(true_fvc)
            all_pred_fvc.extend(fvc_pred.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())

    # Calculate official metric
    val_score = score_function(
        np.array(all_true_fvc), np.array(all_pred_fvc), np.array(all_confidence)
    )

    return val_score


def run_training():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df, batch_size=Config.BATCH_SIZE
    )

    # Initialize Model
    print("Initializing Model...")
    model = DualAxisNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    criterion = CombinedLoss().to(device)

    # Training Loop Variables
    best_score = -float("inf")
    patience_counter = 0

    print("Starting Training...")
    print("-" * 60)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_m_loss, train_a_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_score = validate_one_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(
            f"Train Loss: {train_loss:.6f} (Metric: {train_m_loss:.6f}, Aux: {train_a_loss:.6f})"
        )
        print(f"Val Score: {val_score:.10f}")  # Full precision

        # Early Stopping & Model Saving
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.6f} -> {val_score:.6f}). Saving Model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

        print("-" * 60)

    print(f"Training Complete. Best Validation Score: {best_score:.10f}")
