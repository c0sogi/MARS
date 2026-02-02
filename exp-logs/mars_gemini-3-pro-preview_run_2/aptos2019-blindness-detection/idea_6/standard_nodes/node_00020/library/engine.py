import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import quadratic_weighted_kappa
from library.model import DRModel


def train_one_epoch(model, loader, optimizer, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.MSELoss()

    for _, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).view(-1, 1)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def valid_one_epoch(model, loader, device):
    """
    Validates the model and calculates MSE Loss and Quadratic Weighted Kappa.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_all = []
    targets_all = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)
            batch_size = images.size(0)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds_all.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Flatten arrays
    preds_flat = np.concatenate(preds_all).flatten()
    targets_flat = np.concatenate(targets_all).flatten()

    # Calculate QWK
    # Round continuous predictions to nearest integer and clip to [0, 4]
    preds_rounded = np.rint(preds_flat).clip(0, 4).astype(int)
    targets_int = targets_flat.astype(int)

    qwk = quadratic_weighted_kappa(targets_int, preds_rounded)

    return epoch_loss, qwk


def run_fold(fold, model_config, train_loader, val_loader):
    """
    Orchestrates the training for a single fold, including SWA and Early Stopping.
    """
    print(f"--- Starting Fold {fold} for {model_config['name']} ---")
    device = Config.DEVICE

    # Initialize Model
    model = DRModel(
        model_name=model_config["name"], pretrained=True, checkpoint_path=None
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    # SWA Setup
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
    swa_active = False

    # Early Stopping Setup
    patience = 5
    patience_counter = 0
    best_loss = float("inf")

    # Checkpoint Paths
    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)

    model_name_safe = model_config["checkpoint_prefix"]
    best_model_path = os.path.join(save_dir, f"{model_name_safe}_fold_{fold}_best.pth")
    swa_model_path = os.path.join(save_dir, f"{model_name_safe}_fold_{fold}_swa.pth")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)

        # SWA Logic: Activate in final epochs
        if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()

        # Validation
        val_loss, val_qwk = valid_one_epoch(model, val_loader, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val QWK: {val_qwk}"
        )

        # Save Best Model (Standard)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Early Stopping Check
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Finalize SWA Model if it was active
    if Config.USE_SWA and swa_active:
        print("Finalizing SWA Model...")
        # Update BatchNorm statistics using the training loader
        update_bn(train_loader, swa_model, device=device)

        # Validate SWA Model
        swa_loss, swa_qwk = valid_one_epoch(swa_model, val_loader, device)
        print(f"SWA Results | Loss: {swa_loss} | QWK: {swa_qwk}")

        # Save SWA Model
        torch.save(swa_model.state_dict(), swa_model_path)

    # Cleanup
    del model, swa_model, optimizer, scaler
    torch.cuda.empty_cache()
