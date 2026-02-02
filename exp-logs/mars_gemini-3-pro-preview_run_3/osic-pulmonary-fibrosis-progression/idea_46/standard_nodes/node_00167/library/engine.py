import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    AverageMeter,
    LaplaceNLLLoss,
    laplace_log_likelihood,
    inverse_scale,
    seed_everything,
)
from library.model import BCOSRNet
from library.data import get_dataloaders, get_submission_dataloader


def get_optimizer_and_scheduler(model):
    """
    Configures the optimizer with differential learning rates and the scheduler.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify backbone parameters for lower LR
        if "residual.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX
    )

    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, clinical, targets) in enumerate(loader):
        images = images.to(device)
        clinical = clinical.to(device)
        targets = targets.to(device)

        # Forward pass
        # Output is [Mean_Normalized, Sigma_Normalized]
        preds = model(images, clinical)

        pred_mean_norm = preds[:, 0]
        pred_sigma_norm = preds[:, 1]

        # Calculate loss
        loss = loss_fn(pred_mean_norm, pred_sigma_norm, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def evaluate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    Returns average metric (higher is better) and average loss.
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for images, clinical, targets in loader:
            images = images.to(device)
            clinical = clinical.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(images, clinical)
            pred_mean_norm = preds[:, 0]
            pred_sigma_norm = preds[:, 1]

            # Calculate Loss (for tracking)
            loss = loss_fn(pred_mean_norm, pred_sigma_norm, targets)
            loss_meter.update(loss.item(), images.size(0))

            # Inverse Scale for Metric Calculation
            # We need predictions and targets in ml
            pred_mean_abs, pred_sigma_abs = inverse_scale(
                pred_mean_norm, pred_sigma_norm
            )

            # Target is normalized in Dataset, so inverse scale it too
            target_abs = targets * Config.TARGET_STD + Config.TARGET_MEAN

            # Calculate Metric
            # Note: laplace_log_likelihood expects numpy arrays or handles tensors internally via detach
            score = laplace_log_likelihood(
                target_abs,
                pred_mean_abs,
                pred_sigma_abs,
                clip_sigma=True,
                clip_delta=True,
            )
            metric_meter.update(score, images.size(0))

    return metric_meter.avg, loss_meter.avg


def run_training(debug=False):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")
    if debug:
        print("Debug mode enabled: Using subset of data.")

    # Data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # Model
    model = BCOSRNet()
    model.to(device)

    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    # Loss
    loss_fn = LaplaceNLLLoss()

    # Tracking
    best_metric = -float("inf")
    best_epoch = 0
    patience = 10
    epochs_no_improve = 0

    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )

        # Validate
        val_metric, val_loss = evaluate(model, val_loader, loss_fn, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with metric: {best_metric}")
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print(f"Training complete. Best Metric: {best_metric} at Epoch {best_epoch+1}")
    return best_metric


def generate_submission():
    """
    Generates submission file using the best trained model.
    """
    device = torch.device(Config.DEVICE)
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Skipping submission generation.")
        return

    # Load Model
    model = BCOSRNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Get Data
    loader, sub_df = get_submission_dataloader()

    all_fvc = []
    all_conf = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, clinical, _ in loader:
            images = images.to(device)
            clinical = clinical.to(device)

            preds = model(images, clinical)
            pred_mean_norm = preds[:, 0]
            pred_sigma_norm = preds[:, 1]

            # Inverse Scale
            pred_mean_abs, pred_sigma_abs = inverse_scale(
                pred_mean_norm, pred_sigma_norm
            )

            # Post-processing:
            # 1. Ensure sigma >= 70 (Metric requirement)
            # Although model has softplus + 70 floor, we apply hard max for safety in submission
            pred_sigma_abs = torch.clamp(pred_sigma_abs, min=70.0)

            all_fvc.extend(pred_mean_abs.cpu().numpy())
            all_conf.extend(pred_sigma_abs.cpu().numpy())

    # Update DataFrame
    sub_df["FVC"] = all_fvc
    sub_df["Confidence"] = all_conf

    # Keep only required columns
    submission = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
