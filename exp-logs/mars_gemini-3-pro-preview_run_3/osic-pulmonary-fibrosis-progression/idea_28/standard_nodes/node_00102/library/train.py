import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math

from library.config import Config
from library.utils import seed_everything, calculate_metric, AverageMeter
from library.data import get_dataloaders
from library.model import MACAN


class LaplaceLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2 = Config.SQRT_2

    def forward(self, mu_pred, sigma_pred, target):
        """
        Args:
            mu_pred: Predicted FVC (B,)
            sigma_pred: Predicted Confidence (B,) - assumed positive via Softplus
            target: True FVC (B,)
        """
        # Calculate absolute error
        delta = torch.abs(target - mu_pred)

        # Calculate loss terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2 * delta) / sigma_pred

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * sigma_pred)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device).squeeze(-1)  # Shape (B,)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(images, tabular)

        # Compute loss
        loss = criterion(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets_scaled = batch["target"].to(device).squeeze(-1)

            # Forward pass
            mu_scaled, sigma_scaled = model(images, tabular)

            # Compute Loss (on scaled data)
            loss = criterion(mu_scaled, sigma_scaled, targets_scaled)
            loss_meter.update(loss.item(), images.size(0))

            # --- Inverse Transform for Metric Calculation ---
            # mu_real = mu_scaled * std + mean
            mu_real = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            # sigma_real = sigma_scaled * std
            sigma_real = sigma_scaled * Config.TARGET_STD

            # target_real = target_scaled * std + mean
            targets_real = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            # Calculate Metric
            score = calculate_metric(targets_real, mu_real, sigma_real)
            metric_meter.update(score, images.size(0))

    return loss_meter.avg, metric_meter.avg


def train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # 2. Model
    model = MACAN().to(device)

    # 3. Optimization
    # Differential Learning Rates
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "img_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLoss()

    # 4. Training Loop
    best_metric = -float("inf")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")


def inference():
    """
    Generates predictions for the test set and saves submission.csv.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Model
    model = MACAN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using random weights.")
    model.eval()

    # 2. Load Data
    # We use the test_loader to get processed images and baseline features for each patient
    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=Config.NUM_WORKERS)

    # Load sample submission to know which weeks to predict
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Prepare results list
    results = []

    # 3. Inference Loop
    # We iterate over patients (from test_loader), then expand for all requested weeks
    with torch.no_grad():
        for batch in test_loader:
            patient_id = batch["patient_id"][0]  # Batch size is 1

            # Get baseline data
            image = batch["image"].to(device)  # (1, 3, H, W)
            base_tabular = batch["tabular"].to(device)  # (1, 8)
            base_week = batch["weeks"].item()

            # Find all requested rows for this patient in submission file
            # Format of Patient_Week: ID..._WeekNum
            patient_rows = sub_df[
                sub_df["Patient_Week"].str.startswith(patient_id + "_")
            ]

            if len(patient_rows) == 0:
                continue

            # Extract requested weeks
            # Row format: ID_Week -> split by last underscore
            requested_weeks = (
                patient_rows["Patient_Week"]
                .apply(lambda x: int(x.split("_")[-1]))
                .values
            )

            # Create batch for this patient
            n_samples = len(requested_weeks)

            # Replicate image: (N, 3, H, W)
            batch_images = image.repeat(n_samples, 1, 1, 1)

            # Replicate tabular: (N, 8)
            batch_tabular = base_tabular.repeat(n_samples, 1)

            # Update 'Scaled_Rel_Weeks' (Index 1 in tabular vector)
            # Rel_Weeks = Requested_Week - Base_Week
            # Scaled = Rel_Weeks * TIME_SCALE
            rel_weeks = torch.tensor(
                requested_weeks - base_week, dtype=torch.float32, device=device
            )
            scaled_rel_weeks = rel_weeks * Config.TIME_SCALE

            batch_tabular[:, 1] = scaled_rel_weeks

            # Predict
            mu_scaled, sigma_scaled = model(batch_images, batch_tabular)

            # Inverse Transform
            mu_real = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_real = sigma_scaled * Config.TARGET_STD

            # Collect results
            for i, week in enumerate(requested_weeks):
                pid_week = f"{patient_id}_{week}"
                fvc_pred = mu_real[i].item()
                conf_pred = sigma_real[i].item()

                # Apply final clip for submission consistency (though metric handles it too)
                conf_pred = max(conf_pred, Config.CONFIDENCE_CLIP)

                results.append(
                    {"Patient_Week": pid_week, "FVC": fvc_pred, "Confidence": conf_pred}
                )

    # 4. Save Submission
    submission = pd.DataFrame(results)

    # Ensure strict ordering matching sample_submission if possible,
    # though merging usually handles it. We'll just save what we have.
    # The competition usually expects all rows present.

    # Check if we missed any rows (e.g. patients not in test folder?)
    # We merge with original sample_submission to ensure order and completeness
    final_sub = sub_df[["Patient_Week"]].merge(
        submission, on="Patient_Week", how="left"
    )

    # Fill missing (if any) with defaults
    final_sub["FVC"] = final_sub["FVC"].fillna(2000)
    final_sub["Confidence"] = final_sub["Confidence"].fillna(100)

    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    Config.setup()
    train_model()
    inference()
