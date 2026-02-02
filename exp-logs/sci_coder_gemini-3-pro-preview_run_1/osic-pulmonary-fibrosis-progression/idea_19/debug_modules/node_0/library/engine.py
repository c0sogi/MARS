import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from library.config import Config
from library.utils import laplace_log_likelihood


def criterion(fvc_true, fvc_pred, sigma, device):
    """
    Differentiable implementation of the negative modified Laplace Log Likelihood.
    Used as the loss function for optimization.

    Formula:
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
        loss = -metric
    """
    # Constants
    metric_clip_err = torch.tensor(Config.METRIC_CLIP_ERR, device=device)
    metric_min_conf = torch.tensor(Config.METRIC_MIN_CONF, device=device)
    sqrt_2 = torch.tensor(2.0, device=device).sqrt()

    # 1. Clip the confidence (sigma)
    # Enforce gradient flow through sigma even if clipped locally,
    # though max operation acts as a gate.
    sigma_clipped = torch.max(sigma, metric_min_conf)

    # 2. Calculate the absolute error (delta)
    abs_error = torch.abs(fvc_true - fvc_pred)
    delta = torch.min(abs_error, metric_clip_err)

    # 3. Compute the metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    # 4. Return Loss (Maximize metric -> Minimize negative metric)
    loss = -metric
    return loss.mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, data in enumerate(dataloader):
        # Move inputs to device
        image_axial = data["image_axial"].to(device)
        image_coronal = data["image_coronal"].to(device)
        tabular = data["tabular"].to(device)
        dt = data["dt"].to(device)
        baseline_fvc = data["baseline_fvc"].to(device)
        target = data["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_axial, image_coronal, tabular, dt, baseline_fvc)

        fvc_pred = outputs["fvc_pred"]
        confidence_pred = outputs["confidence_pred"]

        # Calculate loss
        loss = criterion(target, fvc_pred, confidence_pred, device)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set using the official numpy-based metric.
    """
    model.eval()

    all_targets = []
    all_preds = []
    all_sigmas = []

    with torch.no_grad():
        for data in dataloader:
            image_axial = data["image_axial"].to(device)
            image_coronal = data["image_coronal"].to(device)
            tabular = data["tabular"].to(device)
            dt = data["dt"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            target = data["target"].to(device)

            outputs = model(image_axial, image_coronal, tabular, dt, baseline_fvc)

            fvc_pred = outputs["fvc_pred"]
            confidence_pred = outputs["confidence_pred"]

            all_targets.append(target.cpu().numpy())
            all_preds.append(fvc_pred.cpu().numpy())
            all_sigmas.append(confidence_pred.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    # Calculate metric
    score = laplace_log_likelihood(y_true, y_pred, sigma)
    return score


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    print(f"Starting training on device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_score = -float("inf")
    early_stop_counter = 0

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Score: {val_score:.8f}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Score! Model saved to {Config.BEST_MODEL_PATH}")
        else:
            early_stop_counter += 1
            print(f"  >>> No improvement. Patience: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.8f}")
    return best_score
