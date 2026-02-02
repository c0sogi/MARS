import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, metric_score
from library.data import get_dataloaders
from library.model import RaliNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, tabular, targets) in enumerate(loader):
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Output is [FVC_pred, Raw_Sigma]
        preds = model(images, tabular)

        # Calculate loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device, stats):
    """
    Evaluates the model on the validation set.
    Computes Loss (on standardized data) and Metric (on original scale).
    """
    model.eval()
    running_loss = 0.0

    all_targets = []
    all_fvc_preds = []
    all_sigma_preds = []

    # Extract scaling stats
    fvc_std = stats["fvc_std"]
    fvc_mean = stats["fvc_mean"]

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(images, tabular)

            # Loss (Standardized space)
            loss = criterion(preds, targets)
            running_loss += loss.item() * images.size(0)

            # Process predictions for Metric calculation
            # 1. Get standardized FVC and Sigma
            fvc_pred_std = preds[:, 0].cpu().numpy()
            raw_sigma_std = preds[:, 1].cpu().numpy()

            # 2. Convert Sigma: softplus(raw) -> positive
            sigma_pred_std = np.log(1 + np.exp(raw_sigma_std)) + Config.EPSILON

            # 3. Inverse Transform to original scale (ml)
            # FVC_real = FVC_std * std + mean
            fvc_pred_real = fvc_pred_std * fvc_std + fvc_mean
            # Sigma_real = Sigma_std * std (Scale only)
            sigma_pred_real = sigma_pred_std * fvc_std

            # 4. Inverse Transform Targets
            targets_real = targets.cpu().numpy() * fvc_std + fvc_mean

            all_targets.extend(targets_real)
            all_fvc_preds.extend(fvc_pred_real)
            all_sigma_preds.extend(sigma_pred_real)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate Competition Metric
    score = metric_score(all_targets, all_fvc_preds, all_sigma_preds)

    return epoch_loss, score


def generate_submission(model, loader, device, stats):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()

    results = []
    fvc_std = stats["fvc_std"]
    fvc_mean = stats["fvc_mean"]

    with torch.no_grad():
        for images, tabular, _, patient_weeks in loader:
            images = images.to(device)
            tabular = tabular.to(device)

            preds = model(images, tabular)

            fvc_pred_std = preds[:, 0].cpu().numpy()
            raw_sigma_std = preds[:, 1].cpu().numpy()

            # Softplus
            sigma_pred_std = np.log(1 + np.exp(raw_sigma_std)) + Config.EPSILON

            # Inverse Transform
            fvc_pred_real = fvc_pred_std * fvc_std + fvc_mean
            sigma_pred_real = sigma_pred_std * fvc_std

            # Iterate through batch
            for i in range(len(patient_weeks)):
                pw = patient_weeks[i]
                fvc = fvc_pred_real[i]
                sigma = sigma_pred_real[i]

                # Apply submission constraints
                # Note: The metric clips sigma at 70, so we should likely output at least 70
                # or output the raw confidence and let the evaluator clip.
                # However, the task description says: "confidence values are clipped at 70 ml".
                # Usually providing the raw value is safer, but here we enforce the lower bound.
                sigma = max(sigma, 70)

                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": sigma})

    df = pd.DataFrame(results)
    df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training(debug=False):
    """
    Main training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
        load_cached_data=True,
    )

    # 2. Model
    model = RaliNet().to(device)

    # 3. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest (heads/streams)
    backbone_ids = list(map(id, model.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Loss
    criterion = LaplaceLogLikelihoodLoss()

    # 6. Training Loop
    best_metric = -float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = evaluate(model, val_loader, criterion, device, stats)

        # Step Scheduler
        scheduler.step()

        # Print
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Checkpoint
        if val_metric > best_metric:
            print(
                f"Metric improved ({best_metric:.6f} -> {val_metric:.6f}). Saving model..."
            )
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Metric: {best_metric:.10f}")

    # 7. Generate Submission
    # Load best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    generate_submission(model, test_loader, device, stats)
