import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, AverageMeter, metric_laplace_log_likelihood
from library.data import OSICDataset
from library.model import AVRDAN


def train_fn(dataloader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in dataloader:
        # Move inputs to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tab_glu = batch["tab_glu"].to(device)
        tab_skip = batch["tab_skip"].to(device)
        delta_week = batch["delta_week"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, conf_pred = model(
            img_ax, img_cor, tab_glu, tab_skip, delta_week, baseline_fvc
        )

        # Calculate Loss
        # The metric is negative (higher is better).
        # We want to maximize the metric, so we minimize -metric.
        metric_score = metric_laplace_log_likelihood(target, fvc_pred, conf_pred)
        loss = -metric_score

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), img_ax.size(0))

    if scheduler:
        scheduler.step()

    return loss_meter.avg


def eval_fn(dataloader, model, device):
    """
    Evaluates the model on the validation set.
    Returns the average metric score (higher is better).
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab_glu = batch["tab_glu"].to(device)
            tab_skip = batch["tab_skip"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            fvc_pred, conf_pred = model(
                img_ax, img_cor, tab_glu, tab_skip, delta_week, baseline_fvc
            )

            score = metric_laplace_log_likelihood(target, fvc_pred, conf_pred)
            metric_meter.update(score.item(), img_ax.size(0))

    return metric_meter.avg


def run_training():
    """
    Main training loop with early stopping.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")

    # --- Data Loading ---
    train_dataset = OSICDataset(csv_path=Config.TRAIN_CSV, mode="train")
    val_dataset = OSICDataset(csv_path=Config.VAL_CSV, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    model = AVRDAN()
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # --- Training Loop ---
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
        val_score = eval_fn(val_loader, model, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")


def generate_submission():
    """
    Generates predictions for the test set using the best saved model.
    """
    print("Generating submission...")
    seed_everything(Config.SEED)

    # Load Test Data
    test_dataset = OSICDataset(csv_path=Config.TEST_CSV, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = AVRDAN()
    model.to(Config.DEVICE)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
        print("Loaded best model weights.")
    else:
        print(
            "Warning: Best model weights not found. Using random initialization (likely to fail)."
        )

    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(Config.DEVICE)
            img_cor = batch["img_cor"].to(Config.DEVICE)
            tab_glu = batch["tab_glu"].to(Config.DEVICE)
            tab_skip = batch["tab_skip"].to(Config.DEVICE)
            delta_week = batch["delta_week"].to(Config.DEVICE)
            baseline_fvc = batch["baseline_fvc"].to(Config.DEVICE)

            # Metadata
            patient_weeks = batch["patient_week"]

            # Predict
            fvc_pred, conf_pred = model(
                img_ax, img_cor, tab_glu, tab_skip, delta_week, baseline_fvc
            )

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            conf_pred = conf_pred.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, fvc_pred, conf_pred):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create Submission DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save to Config path
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Also save to root submission.csv if needed by environment,
    # though Config points to working/idea_31/submission.csv
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(sub_df.head())
