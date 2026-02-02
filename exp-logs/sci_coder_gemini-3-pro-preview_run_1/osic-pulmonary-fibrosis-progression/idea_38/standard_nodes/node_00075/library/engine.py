import torch
import torch.nn as nn
import math
import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, score_function


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Computes the modified Laplace Log Likelihood loss.
    The competition metric is defined as:
        Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    We minimize the Loss = -Metric.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, alpha, sigma_base, sigma_growth, baseline_fvc, rel_week, true_fvc
    ):
        """
        Args:
            alpha: (B, 1) Slope parameter
            sigma_base: (B, 1) Base uncertainty
            sigma_growth: (B, 1) Uncertainty growth rate
            baseline_fvc: (B, 1) Baseline FVC
            rel_week: (B, 1) Weeks relative to baseline
            true_fvc: (B, 1) Ground truth FVC
        """
        # 1. Calculate Parametric Predictions
        # FVC_pred = Baseline + alpha * (Week - Baseline_Week)
        pred_fvc = baseline_fvc + alpha * rel_week

        # Sigma_pred = Base + Growth * |Week - Baseline_Week|
        pred_sigma = sigma_base + sigma_growth * torch.abs(rel_week)

        # 2. Apply Metric Constraints (for Loss calculation)
        # Clip Confidence at 70 ml
        sigma_clipped = torch.clamp(pred_sigma, min=Config.sigma_clip)

        # Clip Absolute Error at 1000 ml
        error = torch.abs(true_fvc - pred_fvc)
        delta = torch.clamp(error, max=Config.max_fvc_error)

        # 3. Compute Loss terms
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        sqrt_2 = math.sqrt(2)
        loss = (sqrt_2 * delta) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, device, loss_fn):
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
        rel_week = batch["rel_week"].to(device).unsqueeze(1)  # (B, 1)
        true_fvc = batch["fvc"].to(device).unsqueeze(1)  # (B, 1)
        baseline_fvc = batch["baseline_fvc"].to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

        # Compute loss
        loss = loss_fn(
            alpha, sigma_base, sigma_growth, baseline_fvc, rel_week, true_fvc
        )

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update statistics
        loss_meter.update(loss.item(), axial.size(0))

    return loss_meter.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    score_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["rel_week"].to(device).unsqueeze(1)
            true_fvc = batch["fvc"].to(device).unsqueeze(1)
            baseline_fvc = batch["baseline_fvc"].to(device).unsqueeze(1)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Reconstruct predictions
            pred_fvc = baseline_fvc + alpha * rel_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(rel_week)

            # Compute metric (Higher is better)
            score = score_function(true_fvc, pred_fvc, pred_sigma)
            score_meter.update(score.item(), axial.size(0))

    return score_meter.avg


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience
):
    """
    Manages the training process, including early stopping and model checkpointing.
    """
    loss_fn = LaplaceLogLikelihoodLoss()
    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Score: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.model_path)
            patience_counter = 0
            # print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Score: {best_score}")


def predict(model, loader, device):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    model.eval()
    results = []

    print("Generating predictions...")

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["rel_week"].to(device).unsqueeze(1)
            baseline_fvc = batch["baseline_fvc"].to(device).unsqueeze(1)
            patient_week_ids = batch["patient_week_id"]

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Calculate predictions
            pred_fvc = baseline_fvc + alpha * rel_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(rel_week)

            # Collect results
            pred_fvc_np = pred_fvc.cpu().numpy().flatten()
            pred_sigma_np = pred_sigma.cpu().numpy().flatten()

            for pw_id, fvc, sigma in zip(patient_week_ids, pred_fvc_np, pred_sigma_np):
                results.append({"Patient_Week": pw_id, "FVC": fvc, "Confidence": sigma})

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Save to CSV
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
