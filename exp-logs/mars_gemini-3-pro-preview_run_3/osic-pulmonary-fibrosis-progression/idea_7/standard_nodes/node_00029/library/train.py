import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    SUBMISSION_PATH,
    EPOCHS,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
    METRIC_CLIP_SIGMA,
)
from library.utils import seed_everything, metric_laplace_log_likelihood, DataScaler
from library.data import get_dataloaders
from library.model import SAPNet


class LaplaceNLLLoss(nn.Module):
    """
    Differentiable Laplace Negative Log Likelihood Loss.
    L = (sqrt(2) * |y - mu|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceNLLLoss, self).__init__()

    def forward(self, mu, sigma, target):
        """
        Args:
            mu: Predicted FVC (B,)
            sigma: Predicted Confidence (B,)
            target: True FVC (B,)
        """
        # Calculate absolute error
        delta = torch.abs(target - mu)

        # Calculate NLL
        # Note: sigma is guaranteed to be positive via softplus + floor in the model
        loss = (
            torch.sqrt(torch.tensor(2.0).to(delta.device)) * delta
        ) / sigma + torch.log(torch.sqrt(torch.tensor(2.0).to(delta.device)) * sigma)

        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, tab, targets) in enumerate(loader):
        imgs = imgs.to(device)
        tab = tab.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(imgs, tab)

        # Compute loss
        loss = criterion(mu, sigma, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, scaler, device):
    model.eval()

    all_mu = []
    all_sigma = []
    all_targets = []

    with torch.no_grad():
        for imgs, tab, targets in loader:
            imgs = imgs.to(device)
            tab = tab.to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(imgs, tab)

            # Move to CPU
            mu_scaled = mu_scaled.cpu().numpy()
            sigma_scaled = sigma_scaled.cpu().numpy()
            targets_scaled = targets.numpy()

            # Inverse transform to original scale for metric calculation
            mu = scaler.inverse_transform_target(mu_scaled)
            sigma = scaler.inverse_transform_sigma(sigma_scaled)
            targets = scaler.inverse_transform_target(targets_scaled)

            all_mu.extend(mu)
            all_sigma.extend(sigma)
            all_targets.extend(targets)

    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_targets = np.array(all_targets)

    # Calculate metric using the official competition metric function
    score = metric_laplace_log_likelihood(all_targets, all_mu, all_sigma)

    return score


def train_model(epochs=EPOCHS, load_cached_data=True):
    seed_everything()

    # 1. Data Loading
    train_loader, val_loader, _, scaler = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    model = SAPNet().to(DEVICE)

    # 3. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest (head, attention, projectors)
    backbone_params = []
    head_params = []

    # ID parameters by checking if they belong to the backbone module
    backbone_ids = list(map(id, model.backbone.parameters()))

    for name, param in model.named_parameters():
        if id(param) in backbone_ids:
            if param.requires_grad:
                backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )

    # 5. Loss
    criterion = LaplaceNLLLoss()

    # 6. Training Loop
    best_score = -float("inf")

    print(f"Starting training on {DEVICE} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_score = evaluate(model, val_loader, scaler, DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Metric: {val_score:.10f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_means": scaler.means,
                    "scaler_stds": scaler.stds,
                    "best_score": best_score,
                },
                BEST_MODEL_PATH,
            )
            print(f"  >>> New Best Model Saved! Score: {best_score:.10f}")

    print(f"Training complete. Best Validation Score: {best_score:.10f}")
    return scaler


def generate_submission(scaler):
    print("Generating submission...")

    # Load Test Data
    _, _, test_loader, _ = get_dataloaders(load_cached_data=True)

    # Load Best Model
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"No model checkpoint found at {BEST_MODEL_PATH}")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

    # Initialize model and load weights
    model = SAPNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Ensure scaler is fitted with the stats from training
    scaler.means = checkpoint["scaler_means"]
    scaler.stds = checkpoint["scaler_stds"]
    scaler.fitted = True

    results = []

    with torch.no_grad():
        for imgs, tab, patient_week_ids in test_loader:
            imgs = imgs.to(DEVICE)
            tab = tab.to(DEVICE)

            # Predict
            mu_scaled, sigma_scaled = model(imgs, tab)

            # Inverse Transform
            mu = scaler.inverse_transform_target(mu_scaled.cpu().numpy())
            sigma = scaler.inverse_transform_sigma(sigma_scaled.cpu().numpy())

            # Post-processing for submission
            # Clip sigma to 70 as per metric definition
            sigma = np.maximum(sigma, METRIC_CLIP_SIGMA)

            for pw, fvc, conf in zip(patient_week_ids, mu, sigma):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create DataFrame and Save
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(sub_df.head())
