import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    inverse_transform,
    calculate_metric,
    print_metrics,
)
from library.data import get_dataloaders, prepare_submission_dataframe
from library.model import CASDSNet


class LLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss for Standardized Targets.

    Formula:
        L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LLLLoss, self).__init__()

    def forward(self, mu, sigma, target):
        """
        Args:
            mu (Tensor): Predicted mean (standardized).
            sigma (Tensor): Predicted std (standardized).
            target (Tensor): Ground truth (standardized).
        """
        # Calculate absolute error
        delta = torch.abs(target - mu)

        # Calculate Loss terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (torch.sqrt(torch.tensor(2.0).to(delta.device)) * delta) / sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(torch.sqrt(torch.tensor(2.0).to(delta.device)) * sigma)

        loss = term1 + term2
        return torch.mean(loss)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, tabular, targets in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device).squeeze(-1)  # Ensure shape [Batch]

        optimizer.zero_grad()

        mu, sigma = model(images, tabular)

        loss = criterion(mu, sigma, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_mu = []
    all_sigma = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device).squeeze(-1)

            mu, sigma = model(images, tabular)

            loss = criterion(mu, sigma, targets)
            running_loss += loss.item() * images.size(0)

            # Store for metric calculation (inverse transform needed)
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_targets.append(targets)

    # Calculate Loss
    avg_loss = running_loss / len(loader.dataset)

    # Calculate Metric on Original Scale
    all_mu = torch.cat(all_mu)
    all_sigma = torch.cat(all_sigma)
    all_targets = torch.cat(all_targets)

    # Inverse transform to ml
    mu_orig, sigma_orig = inverse_transform(all_mu, all_sigma)

    # Inverse transform targets: target_orig = target_std * std + mean
    target_orig = all_targets.cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN

    # Calculate competition metric
    score = calculate_metric(target_orig, mu_orig, sigma_orig)

    return avg_loss, score


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission...")

    # Get test loader (shuffle=False is crucial here)
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    model.eval()
    predictions_mu = []
    predictions_sigma = []

    with torch.no_grad():
        for images, tabular, _ in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            mu, sigma = model(images, tabular)

            # Inverse transform immediately
            mu_orig, sigma_orig = inverse_transform(mu, sigma)

            predictions_mu.extend(mu_orig.tolist())
            predictions_sigma.extend(sigma_orig.tolist())

    # Load sample submission to get IDs and structure
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Ensure lengths match
    if len(sub_df) != len(predictions_mu):
        print(
            f"Warning: Length mismatch. Submission DF: {len(sub_df)}, Preds: {len(predictions_mu)}"
        )
        # In case of mismatch (e.g. debug mode), truncate or pad would be needed,
        # but with correct loaders they should match.
        sub_df = sub_df.iloc[: len(predictions_mu)]

    # Assign predictions
    sub_df["FVC"] = predictions_mu
    sub_df["Confidence"] = predictions_sigma

    # Post-processing: Clip Confidence at 70ml
    sub_df["Confidence"] = sub_df["Confidence"].apply(lambda x: max(x, 70))

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training():
    seed_everything(Config.SEED)

    # 1. Data Setup
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    # 2. Model Setup
    device = torch.device(Config.DEVICE)
    model = CASDSNet().to(device)

    # 3. Optimization Setup
    # Differential Learning Rates
    backbone_params = list(model.image_encoder.parameters())
    # All other parameters (heads, streams)
    backbone_ids = list(map(id, backbone_params))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)
    criterion = LLLLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    best_score = float("-inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_score = val_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print_metrics("Train", epoch + 1, train_loss, 0.0)  # Score not calc for train
        print_metrics("Val", epoch + 1, val_loss, val_score)

        # Early Stopping & Checkpointing
        # We monitor Validation Loss (Standardized LLL) as the primary objective
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(
        f"Training complete. Best Val Loss: {best_val_loss:.6f}, Best Score: {best_score:.6f}"
    )

    # 5. Inference
    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, device)


if __name__ == "__main__":
    # This block is kept for local testing if run directly,
    # but the function run_training is the entry point.
    run_training()
