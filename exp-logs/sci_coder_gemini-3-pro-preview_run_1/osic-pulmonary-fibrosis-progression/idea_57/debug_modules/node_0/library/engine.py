import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import laplace_log_likelihood_metric


class LaplaceLoss(nn.Module):
    """
    Computes the Negative Log Likelihood (NLL) for a Laplace distribution,
    adapted for the parametric outputs of the SCSL-Net.

    The model predicts parameters alpha, sigma_base, and sigma_growth.
    Predictions are calculated as:
        FVC_pred = Baseline_FVC + alpha * (Weeks - Baseline_Week)
        Sigma_pred = sigma_base + sigma_growth * |Weeks - Baseline_Week|

    Loss = (sqrt(2) * |True - Pred|) / Sigma + ln(sqrt(2) * Sigma)
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()
        self.device = Config.DEVICE

    def forward(self, preds, target, weeks, baseline_weeks, baseline_fvc):
        """
        Args:
            preds: (B, 3) tensor [alpha, sigma_base, sigma_growth]
            target: (B,) tensor [True FVC]
            weeks: (B,) tensor [Current Week]
            baseline_weeks: (B,) tensor [Baseline Week]
            baseline_fvc: (B,) tensor [Baseline FVC]
        """
        # Unpack predictions
        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # Calculate time delta
        dt = weeks - baseline_weeks

        # Calculate predicted FVC and Sigma based on trajectory
        fvc_pred = baseline_fvc + alpha * dt
        sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

        # Clip sigma to avoid instability and match metric definition
        # We use the competition floor of 70ml
        sigma_clipped = torch.clamp(sigma_pred, min=Config.CONFIDENCE_CLIP)

        # Calculate Absolute Error
        delta = torch.abs(target - fvc_pred)

        # Note: We do NOT clip delta at 1000ml for the Loss function.
        # Clipping error in loss kills gradients for bad predictions.
        # We only clip sigma.

        # Calculate NLL
        # Metric formula term: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # We want to MAXIMIZE metric, so we MINIMIZE negative metric (Loss).
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=self.device))
        loss = (sqrt_2 * delta) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

        return loss.mean()


def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    criterion = LaplaceLoss()

    # Iterate over batches
    for batch_idx, data in enumerate(loader):
        # Move data to device
        img_axial = data["image_axial"].to(device)
        img_coronal = data["image_coronal"].to(device)
        tabular = data["tabular"].to(device)
        target = data["target"].to(device)

        # Meta data for parametric calculation
        weeks = data["meta"]["Weeks"].to(device)
        base_weeks = data["meta"]["Baseline_Week"].to(device)
        base_fvc = data["meta"]["Baseline_FVC"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model outputs static parameters: [alpha, sigma_base, sigma_growth]
        preds = model(img_axial, img_coronal, tabular)

        # Compute Loss
        loss = criterion(preds, target, weeks, base_weeks, base_fvc)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Note: Scheduler step is typically handled in the main loop for Epoch-based schedulers
    # (like CosineAnnealingLR defined in Config).

    avg_loss = running_loss / len(loader)
    return avg_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for data in loader:
            img_axial = data["image_axial"].to(device)
            img_coronal = data["image_coronal"].to(device)
            tabular = data["tabular"].to(device)
            target = data["target"].to(device)

            weeks = data["meta"]["Weeks"].to(device)
            base_weeks = data["meta"]["Baseline_Week"].to(device)
            base_fvc = data["meta"]["Baseline_FVC"].to(device)

            # Forward pass
            preds = model(img_axial, img_coronal, tabular)

            # Unpack and calculate trajectory
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            dt = weeks - base_weeks

            fvc_pred = base_fvc + alpha * dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            # Calculate metric for this batch
            # Note: utils.laplace_log_likelihood_metric handles the clipping logic internally
            # (sigma clipped at 70, error clipped at 1000)
            batch_score = laplace_log_likelihood_metric(target, fvc_pred, sigma_pred)
            scores.append(batch_score)

    return np.mean(scores)
