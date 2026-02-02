import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import ZIOSRNet


class LaplaceNLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    Optimizes the NLL of a Laplace distribution with sqrt(2) scaling.
    L = (sqrt(2) * |y_true - y_pred|) / sigma + log(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceNLLLoss, self).__init__()
        self.sqrt_2 = Config.METRIC_CONST_SQRT2

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        # Calculate absolute error
        delta = torch.abs(true_fvc - pred_fvc)

        # Calculate NLL terms
        # Note: pred_sigma is guaranteed positive via Softplus in the model
        term1 = (self.sqrt_2 * delta) / pred_sigma
        term2 = torch.log(self.sqrt_2 * pred_sigma)

        loss = torch.mean(term1 + term2)
        return loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, features, targets in loader:
        images = images.to(device)
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        mu_pred, sigma_pred = model(images, features)

        # Compute loss
        loss = criterion(mu_pred, sigma_pred, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device, stats):
    model.eval()
    all_true_fvc = []
    all_pred_fvc = []
    all_pred_sigma = []

    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    with torch.no_grad():
        for images, features, targets in loader:
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass (Normalized outputs)
            mu_norm, sigma_norm = model(images, features)

            # Inverse Transform to original scale (ml)
            # mu_real = mu_norm * std + mean
            # sigma_real = sigma_norm * std
            pred_fvc_real = mu_norm * fvc_std + fvc_mean
            pred_sigma_real = sigma_norm * fvc_std

            # Inverse Transform targets
            true_fvc_real = targets * fvc_std + fvc_mean

            all_true_fvc.extend(true_fvc_real.cpu().numpy())
            all_pred_fvc.extend(pred_fvc_real.cpu().numpy())
            all_pred_sigma.extend(pred_sigma_real.cpu().numpy())

    # Compute metric using the library utility
    metric_score = compute_metric(
        np.array(all_true_fvc), np.array(all_pred_fvc), np.array(all_pred_sigma)
    )

    return metric_score


def generate_submission(model, loader, device, stats):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()
    results = []

    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    # Access dataset metadata to reconstruct Patient_Week IDs
    # The loader is not shuffled, so order is preserved
    dataset_data = loader.dataset.data
    current_idx = 0

    with torch.no_grad():
        for images, features, _ in loader:
            images = images.to(device)
            features = features.to(device)

            batch_size = images.size(0)

            # Predict
            mu_norm, sigma_norm = model(images, features)

            # Inverse Transform
            pred_fvc_real = mu_norm.cpu().numpy() * fvc_std + fvc_mean
            pred_sigma_real = sigma_norm.cpu().numpy() * fvc_std

            for i in range(batch_size):
                # Get metadata for this sample
                meta = dataset_data[current_idx + i]
                patient = meta["patient"]
                week = meta["weeks"]

                # Format ID
                patient_week = f"{patient}_{week}"

                # Post-processing
                fvc = pred_fvc_real[i]
                # Clip confidence as per task requirement (min 70)
                conf = max(pred_sigma_real[i], 70)

                results.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": conf}
                )

            current_idx += batch_size

    # Save to CSV
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_experiment(load_cached_data=True):
    """
    Main driver function to run the training and submission pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Retrieve normalization stats from the dataset
    stats = train_loader.dataset.stats

    # 2. Initialize Model
    print("Initializing ZI-OSR Net...")
    model = ZIOSRNet().to(device)

    # 3. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest (heads + anchor)
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "residual.backbone" in name:
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

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Loss
    criterion = LaplaceNLLLoss()

    # 6. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience = 10
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metric = validate(model, val_loader, device, stats)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Loss: {train_loss:.6f} | "
            f"Val Metric: {val_metric:.8f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Metric! Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.8f}")

    # 7. Generate Submission
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device, stats)
