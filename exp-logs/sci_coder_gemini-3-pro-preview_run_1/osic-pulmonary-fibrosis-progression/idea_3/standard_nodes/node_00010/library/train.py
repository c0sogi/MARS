import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, AverageMeter, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import DualAxisTriSlabModel


# ==========================================
# Helper: Trajectory Prediction Logic
# ==========================================
def predict_trajectory(outputs, base_fvc, time_delta):
    """
    Decodes model outputs into FVC and Confidence predictions.
    Ensures consistency between Training (Loss) and Inference (Submission).

    Args:
        outputs: (B, 3) tensor [alpha, sigma_base, sigma_growth]
        base_fvc: (B,) tensor
        time_delta: (B,) tensor

    Returns:
        pred_fvc: (B,) tensor
        pred_sigma: (B,) tensor
    """
    alpha = outputs[:, 0]
    sigma_base = outputs[:, 1]
    sigma_growth = outputs[:, 2]

    # FVC Prediction: Linear decline from baseline
    pred_fvc = base_fvc + alpha * time_delta

    # Confidence Prediction:
    # 1. Use Softplus to ensure positive contributions from the network
    # 2. Add 70 (Config.MIN_CONFIDENCE) to ensure we are always above the clipping threshold
    #    This prevents zero gradients which would occur if we relied solely on the metric's clip.
    pred_sigma = (
        Config.MIN_CONFIDENCE
        + F.softplus(sigma_base)
        + F.softplus(sigma_growth) * torch.abs(time_delta)
    )

    return pred_fvc, pred_sigma


# ==========================================
# Loss Function
# ==========================================
class ParametricLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, base_fvc, time_delta):
        """
        Calculates the negative Laplace Log Likelihood.
        """
        pred_fvc, pred_sigma = predict_trajectory(outputs, base_fvc, time_delta)

        # Calculate metric (Higher is better, so Loss is negative metric)
        metric = laplace_log_likelihood(targets, pred_fvc, pred_sigma)
        return -metric


# ==========================================
# Training & Validation Loops
# ==========================================
def train_one_epoch(epoch, model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    meter = AverageMeter()

    for batch_idx, batch in enumerate(loader):
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        time_delta = batch["time_delta"].to(device)

        # Forward pass
        outputs = model(axial, coronal, tabular)

        # Calculate loss
        loss = criterion(outputs, targets, base_fvc, time_delta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics (Loss = -Metric)
        meter.update(-loss.item(), axial.size(0))

    return meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            time_delta = batch["time_delta"].to(device)

            outputs = model(axial, coronal, tabular)

            # We want the metric value directly
            pred_fvc, pred_sigma = predict_trajectory(outputs, base_fvc, time_delta)
            metric = laplace_log_likelihood(targets, pred_fvc, pred_sigma)

            meter.update(metric.item(), axial.size(0))

    return meter.avg


# ==========================================
# Submission Generation
# ==========================================
def generate_submission(model, loader, device):
    model.eval()
    results = []

    print("Generating predictions for submission...")
    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            time_delta = batch["time_delta"].to(device)

            # Metadata is a dict of lists in the batch
            patient_weeks = batch["meta"]["patient_week"]

            outputs = model(axial, coronal, tabular)
            pred_fvc, pred_sigma = predict_trajectory(outputs, base_fvc, time_delta)

            # Collect results
            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, pred_fvc_np, pred_sigma_np):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure columns are correct
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# ==========================================
# Main Runner
# ==========================================
def run_training():
    seed_everything(Config.SEED)

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 2. Model
    print("Initializing Model...")
    model = DualAxisTriSlabModel(
        backbone_name="efficientnet_b0",
        pretrained=True,
        tabular_input_dim=5,  # Age, Sex, 3xSmoking
        output_dim=3,  # alpha, sigma_base, sigma_growth
    )
    model = model.to(Config.DEVICE)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = ParametricLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_metric = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_metric = validate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Metric: {train_metric:.6f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! (Metric: {best_metric:.6f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {early_stop_counter} epochs without improvement."
                )
                break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    generate_submission(model, test_loader, Config.DEVICE)
    print("Done.")
