import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import SBPDSNet


class LaplaceNLLLoss(nn.Module):
    """
    Differentiable Laplace Negative Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y - pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceNLLLoss, self).__init__()
        self.sqrt_2 = Config.SQRT_2

    def forward(self, pred_mean, pred_sigma, target):
        # pred_sigma is already ensured to be positive via Softplus in the model

        # Absolute error
        delta = torch.abs(target - pred_mean)

        # Calculate NLL terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2 * delta) / pred_sigma

        # Term 2: ln(sqrt(2) * sigma)
        # Note: ln(a*b) = ln(a) + ln(b)
        # ln(sqrt(2)) is constant, but we include it to match the metric definition exactly
        term2 = torch.log(self.sqrt_2 * pred_sigma)

        loss = term1 + term2

        return torch.mean(loss)


def train_epoch(model, loader, optimizer, criterion, device, scalers):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    # Extract scalers for on-the-fly target normalization
    fvc_mean = scalers["fvc_mean"]
    fvc_std = scalers["fvc_std"]

    # Convert scalers to tensors for efficient broadcasting if needed,
    # but simple scalar arithmetic works fine with tensors.

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets_raw = batch["target"].to(device)

        # Normalize targets: (y - mean) / std
        targets_norm = (targets_raw - fvc_mean) / fvc_std

        optimizer.zero_grad()

        outputs = model(images, tabular)

        # Unpack outputs
        final_mean = outputs["final_mean"]
        final_sigma = outputs["final_sigma"]
        base_mean = outputs["base_mean"]
        base_sigma = outputs["base_sigma"]

        # Compute Loss
        # Primary Loss
        loss_final = criterion(final_mean, final_sigma, targets_norm)

        # Auxiliary Loss (Supervision on Stream A)
        loss_base = criterion(base_mean, base_sigma, targets_norm)

        # Total Loss
        loss = loss_final + Config.AUX_LOSS_WEIGHT * loss_base

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device, scalers):
    """
    Evaluates the model on the validation set using the official metric.
    """
    model.eval()

    all_targets = []
    all_preds = []
    all_sigmas = []

    fvc_mean = scalers["fvc_mean"]
    fvc_std = scalers["fvc_std"]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets_raw = batch["target"]  # Keep on CPU for metric calc

            outputs = model(images, tabular)

            # Get normalized predictions
            pred_mean_norm = outputs["final_mean"].cpu().numpy()
            pred_sigma_norm = outputs["final_sigma"].cpu().numpy()

            # Inverse Transform to Raw Scale
            # Mean: norm * std + mean
            pred_mean_raw = pred_mean_norm * fvc_std + fvc_mean

            # Sigma: norm * std (Scaling only)
            pred_sigma_raw = pred_sigma_norm * fvc_std

            all_targets.append(targets_raw.numpy())
            all_preds.append(pred_mean_raw)
            all_sigmas.append(pred_sigma_raw)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_sigmas = np.concatenate(all_sigmas)

    # Calculate official metric
    metric_score = calculate_metric(all_targets, all_preds, all_sigmas)

    return metric_score


def train_model(
    epochs=Config.EPOCHS,
    max_train_samples=Config.MAX_TRAIN_SAMPLES,
    max_val_samples=Config.MAX_VAL_SAMPLES,
):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Setup directories
    checkpoint_dir = os.path.join(Config.CACHE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    # Scalers for normalization/inverse-normalization
    scalers = train_loader.dataset.scalers

    # Model
    model = SBPDSNet().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": model.img_projector.parameters(), "lr": Config.LR_HEAD},
            {"params": model.stream_a.parameters(), "lr": Config.LR_HEAD},
            {"params": model.stream_b_mlp.parameters(), "lr": Config.LR_HEAD},
            {"params": model.stream_b_out.parameters(), "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Loss
    criterion = LaplaceNLLLoss()

    # Training Loop
    best_metric = -float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scalers
        )
        val_metric = validate(model, val_loader, device, scalers)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.10f}"
        )

        # Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            # print(f"  New best model saved! Metric: {best_metric:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")
    return best_model_path, scalers


def generate_submission(model_path, scalers):
    """
    Generates the submission file using the best trained model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    # Note: We don't limit samples for submission
    _, _, test_loader = get_dataloaders(
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        max_train_samples=None,
        max_val_samples=None,
    )

    # Load Model
    model = SBPDSNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    fvc_mean = scalers["fvc_mean"]
    fvc_std = scalers["fvc_std"]

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            patient_weeks = batch["patient_week"]

            outputs = model(images, tabular)

            # Get normalized predictions
            pred_mean_norm = outputs["final_mean"].cpu().numpy()
            pred_sigma_norm = outputs["final_sigma"].cpu().numpy()

            # Inverse Transform
            pred_mean_raw = pred_mean_norm * fvc_std + fvc_mean
            pred_sigma_raw = pred_sigma_norm * fvc_std

            # Post-processing clip for submission
            # "confidence values are clipped at 70 ml"
            pred_sigma_clipped = np.maximum(pred_sigma_raw, Config.MIN_CONFIDENCE)

            for pw, fvc, conf in zip(patient_weeks, pred_mean_raw, pred_sigma_clipped):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
