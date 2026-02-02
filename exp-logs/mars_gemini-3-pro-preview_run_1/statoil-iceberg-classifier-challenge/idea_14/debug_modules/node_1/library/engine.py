import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from library.config import Config
from library.utils import log_message, log_metrics, save_model


def smooth_labels(targets, smoothing=0.0):
    """
    Applies label smoothing to binary targets.
    y_ls = y * (1 - alpha) + 0.5 * alpha
    """
    if smoothing <= 0.0:
        return targets
    with torch.no_grad():
        return targets * (1.0 - smoothing) + 0.5 * smoothing


def custom_update_bn(loader, model, device=None):
    """
    Custom implementation of update_bn that handles multi-input models (image + angle).
    Recalculates Batch Normalization statistics by running a forward pass on the training data.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    with torch.no_grad():
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            # Custom Forward Pass with both inputs
            model(images, angles)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def train_one_epoch(model, loader, optimizer, criterion, device, label_smoothing=0.0):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        batch_size = images.size(0)

        # Apply Label Smoothing
        targets = smooth_labels(labels, label_smoothing)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # For accuracy calculation
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            batch_size = images.size(0)

            logits = model(images, angles)
            loss = criterion(logits, labels)  # No smoothing for validation metric

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct_preds += (preds == labels).sum().item()
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            total_preds += batch_size

    avg_loss = running_loss / dataset_size
    accuracy = correct_preds / total_preds

    return avg_loss, accuracy


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a dictionary mapping IDs to probabilities.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for i, img_id in enumerate(ids):
                results[img_id] = float(probs[i])

    return results


def run_swa_training(
    model,
    train_loader,
    val_loader,
    device=Config.DEVICE,
    swa_start_epoch=Config.SWA_START_EPOCH,
    total_epochs=Config.TOTAL_EPOCHS,
    patience=10,
    save_path_best=None,
    save_path_swa=None,
):
    """
    Orchestrates the Two-Phase training:
    1. Standard Training with CosineAnnealingLR (Epochs 1 to swa_start_epoch).
    2. SWA Training (Epochs swa_start_epoch+1 to total_epochs).

    Handles Early Stopping during Phase 1.
    """
    if save_path_best is None:
        save_path_best = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if save_path_swa is None:
        save_path_swa = os.path.join(Config.CHECKPOINT_DIR, "swa_model.pth")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler for Phase 1
    # We want it to anneal down to a small value by the time SWA starts
    scheduler = CosineAnnealingLR(optimizer, T_max=swa_start_epoch, eta_min=1e-5)

    # SWA Components (Initialized later)
    swa_model = None
    swa_scheduler = None

    best_val_loss = float("inf")
    patience_counter = 0

    log_message(f"Starting training on device: {device}")
    log_message(f"Phase 1: Standard Training (Epochs 1-{swa_start_epoch})")

    model.to(device)

    for epoch in range(1, total_epochs + 1):
        # -----------------------------------------------------------
        # Phase 1: Standard Training
        # -----------------------------------------------------------
        if epoch <= swa_start_epoch:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                label_smoothing=Config.LABEL_SMOOTHING,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            log_metrics(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "phase": "Standard",
                },
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, save_path_best)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log_message(
                    f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss}"
                )
                # If we stop early in Phase 1, we likely shouldn't proceed to SWA blindly.
                # However, for this pipeline, we will load the best model and return.
                # SWA requires a trajectory.
                log_message("Skipping SWA phase due to early stopping.")
                return

        # -----------------------------------------------------------
        # Phase 2: SWA Training
        # -----------------------------------------------------------
        else:
            # Initialize SWA on first epoch of Phase 2
            if swa_model is None:
                log_message(f"Phase 2: SWA Training (Epochs {epoch}-{total_epochs})")

                # Load best model from Phase 1 to ensure good starting point
                # (Optional, but recommended if Phase 1 diverged at the end)
                if os.path.exists(save_path_best):
                    state_dict = torch.load(save_path_best, map_location=device)
                    model.load_state_dict(state_dict)

                swa_model = AveragedModel(model).to(device)

                # SWA Scheduler: Constant high LR or cyclic
                # Using the base LR from config is a common strategy for SWA exploration
                swa_scheduler = SWALR(optimizer, swa_lr=Config.LEARNING_RATE / 2.0)

            # Train standard model (SGD trajectory)
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                label_smoothing=Config.LABEL_SMOOTHING,
            )

            # Update SWA Model
            swa_model.update_parameters(model)
            swa_scheduler.step()

            # We can validate the current model to monitor progress,
            # but the SWA model itself is usually evaluated at the end.
            # Here we validate the underlying model.
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            log_metrics(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,  # Loss of the current stochastic model
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "phase": "SWA",
                },
            )

    # -----------------------------------------------------------
    # Finalize SWA
    # -----------------------------------------------------------
    if swa_model is not None:
        log_message("Updating SWA Batch Normalization statistics...")
        custom_update_bn(train_loader, swa_model, device=device)

        # Validate SWA Model
        swa_val_loss, swa_val_acc = validate(swa_model, val_loader, criterion, device)
        log_message(
            f"Final SWA Results | Val Loss: {swa_val_loss} | Val Acc: {swa_val_acc}"
        )

        save_model(swa_model, save_path_swa)
        log_message(f"SWA model saved to {save_path_swa}")
    else:
        # Fallback if SWA didn't run (e.g. total_epochs <= swa_start_epoch)
        log_message("SWA phase did not run. Best standard model was saved.")
