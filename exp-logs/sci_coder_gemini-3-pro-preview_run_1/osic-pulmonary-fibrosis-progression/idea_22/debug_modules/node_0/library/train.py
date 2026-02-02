import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import prepare_data
from library.model import CASDAN, LaplaceLogLikelihoodLoss


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training using the CAS-DAN architecture.
    Calculates loss based on the parametric trajectory prediction.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        ax = batch["axial"].to(device)
        cor = batch["coronal"].to(device)
        tab = batch["tabular"].to(device)
        target = batch["target"].to(device)
        weeks = batch["week"].to(device)

        optimizer.zero_grad()

        # Forward pass: Model returns parameters for the trajectory
        # alpha (Slope), sigma_base (Intercept Confidence), sigma_growth (Slope Confidence)
        alpha, sigma_base, sigma_growth = model(ax, cor, tab)

        # Reconstruct Baseline FVC from normalized tabular data
        # Feature index 6 is BaseFVC. Normalization was (x - 2500) / 1000
        # Tabular structure: Age(0), Sex(1), Smk(2,3,4), Percent(5), BaseFVC(6)
        base_fvc_rec = tab[:, 6] * 1000.0 + 2500.0

        # Predict FVC: Base + alpha * delta_t
        # For training, delta_t is 'weeks' (relative to baseline)
        fvc_pred = base_fvc_rec + alpha * weeks

        # Predict Confidence: Base + Growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

        # Calculate loss (Negative Log Likelihood approximation)
        loss = criterion(fvc_pred, sigma_pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    val_preds = []
    val_sigmas = []
    val_targets = []

    with torch.no_grad():
        for batch in loader:
            ax = batch["axial"].to(device)
            cor = batch["coronal"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["week"].to(device)

            alpha, sigma_base, sigma_growth = model(ax, cor, tab)

            # Reconstruct Baseline FVC
            base_fvc_rec = tab[:, 6] * 1000.0 + 2500.0

            # Predict
            fvc_pred = base_fvc_rec + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            val_preds.extend(fvc_pred.cpu().numpy())
            val_sigmas.extend(sigma_pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    # Calculate metric using the official evaluation function
    score = get_score(val_targets, val_preds, val_sigmas)
    return score


def run_training():
    """
    Orchestrates the training pipeline: data loading, model setup,
    training loop, validation, and early stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # prepare_data handles caching logic internally
    train_dataset = prepare_data("train", load_cached_data=True)
    val_dataset = prepare_data("val", load_cached_data=True)

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

    # 3. Model Initialization
    model = CASDAN().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_score = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Score: {best_score}")
