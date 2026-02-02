import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.data import OSICDataset, get_scalers
from library.model import OSICModel
from library.utils import seed_everything, AverageMeter, laplace_log_likelihood_metric


class LaplaceNLLLoss(nn.Module):
    """
    Negative Log Likelihood Loss for Laplace Distribution.
    Minimizing this is equivalent to maximizing the Laplace Log Likelihood.

    Loss = (sqrt(2) * |y - mu|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, preds, targets):
        # preds: (B, 2) -> [mu, raw_sigma_score]
        # targets: (B, 1) -> [true_value]

        mu = preds[:, 0]
        # Ensure sigma is positive using softplus
        sigma = F.softplus(preds[:, 1]) + self.eps

        y_true = targets.squeeze()

        # Calculate NLL
        # Term 1: (sqrt(2) * |y - mu|) / sigma
        term1 = (self.sqrt_2.to(mu.device) * torch.abs(y_true - mu)) / sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2.to(mu.device) * sigma)

        loss = torch.mean(term1 + term2)
        return loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        preds = model(images, tabular)
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device, scalers):
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    # Unpacking scalers for metric calculation
    fvc_mean = scalers["fvc_mean"]
    fvc_std = scalers["fvc_std"]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)  # These are scaled targets

            preds = model(images, tabular)

            # 1. Compute Loss (on scaled data)
            loss = criterion(preds, targets)
            loss_meter.update(loss.item(), images.size(0))

            # 2. Compute Metric (on unscaled data, as per competition rules)
            # Unscale Predictions
            mu_scaled = preds[:, 0]
            sigma_scaled = F.softplus(preds[:, 1])

            mu_ml = mu_scaled * fvc_std + fvc_mean
            sigma_ml = sigma_scaled * fvc_std

            # Unscale Targets
            targets_ml = targets.squeeze() * fvc_std + fvc_mean

            # Calculate Metric
            score = laplace_log_likelihood_metric(targets_ml, mu_ml, sigma_ml)
            metric_meter.update(score.item(), images.size(0))

    return loss_meter.avg, metric_meter.avg


def run_training(debug=False, epochs=Config.EPOCHS, load_cached_data=True):
    seed_everything(Config.SEED)
    Config.setup()

    device = Config.DEVICE
    print(f"Training on device: {device}")

    # --- Data Loading ---
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        print(f"DEBUG Mode: Truncating data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        epochs = 2  # Short run for debug

    # Compute scalers from training data
    print("Computing scalers...")
    scalers = get_scalers(train_df)

    # Create Datasets
    train_dataset = OSICDataset(
        train_df, split_type="train", scalers=scalers, load_cached_data=load_cached_data
    )
    val_dataset = OSICDataset(
        val_df, split_type="val", scalers=scalers, load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    print("Initializing model...")
    model = OSICModel().to(device)

    criterion = LaplaceNLLLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # --- Training Loop ---
    best_metric = -float("inf")
    best_epoch = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device, scalers)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

    print(f"Training complete. Best Metric: {best_metric:.6f} at Epoch {best_epoch}")

    # --- Inference on Test Set (Optional but recommended for pipeline completeness) ---
    # Note: Actual submission generation usually happens in a separate script or function,
    # but based on requirements, we might need to verify the model works.
    # We will just verify the best model loads correctly.
    print("Verifying best model checkpoint...")
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()
    print("Verification successful.")
