import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import pandas as pd
import numpy as np
import torch.nn.functional as F

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data import get_loaders
from library.model_factory import create_model, set_backbone_trainable
from library.calibration import TemperatureScaler

logger = get_logger(name="trainer")


class Trainer:
    """
    Handles the training and validation logic for a single epoch.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if Config.USE_AMP else None

    def train_one_epoch(self, loader, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            if Config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def valid_one_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # storage for calibration if needed, though usually done separately
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy


def fit_model(model_name, train_loader, val_loader):
    """
    Runs the full training lifecycle for a specific model architecture,
    implementing Two-Phase training and SWA.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 1. Initialize Model
    model = create_model(model_name).to(device)
    trainer = Trainer(model, device)

    best_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
    swa_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_swa.pth")

    # ==========================
    # Phase 1: Frozen Backbone
    # ==========================
    logger.info(
        f"[{model_name}] Starting Phase 1: Frozen Backbone for {Config.FREEZE_BACKBONE_EPOCHS} epochs."
    )
    set_backbone_trainable(model, False)

    # Optimizer for Head only
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR_HEAD,
        weight_decay=Config.WEIGHT_DECAY,
    )

    for epoch in range(Config.FREEZE_BACKBONE_EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader, optimizer, epoch)
        val_loss, val_acc = trainer.valid_one_epoch(val_loader)
        logger.info(
            f"[{model_name}] Phase 1 Epoch {epoch+1}/{Config.FREEZE_BACKBONE_EPOCHS} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

    # ==========================
    # Phase 2: Unfrozen Backbone + SWA
    # ==========================
    logger.info(
        f"[{model_name}] Starting Phase 2: Fine-tuning for {Config.EPOCHS - Config.FREEZE_BACKBONE_EPOCHS} epochs."
    )
    set_backbone_trainable(model, True)

    # Differential Learning Rates
    # Identify head parameters
    head_ptr = model.get_classifier()
    head_params = list(head_ptr.parameters())
    head_ids = list(map(id, head_params))
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    # SWA Setup
    swa_model = AveragedModel(model) if Config.USE_SWA else None
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR) if Config.USE_SWA else None

    for epoch in range(Config.FREEZE_BACKBONE_EPOCHS, Config.EPOCHS):
        # Train
        train_loss = trainer.train_one_epoch(train_loader, optimizer, epoch)

        # SWA Logic
        is_swa_epoch = Config.USE_SWA and epoch >= Config.SWA_START_EPOCH
        if is_swa_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr_curr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            lr_curr = scheduler.get_last_lr()[0]

        # Validation
        val_loss, val_acc = trainer.valid_one_epoch(val_loader)

        logger.info(
            f"[{model_name}] Phase 2 Epoch {epoch+1}/{Config.EPOCHS} - "
            f"LR: {lr_curr:.2e} - Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Save Best Model (Standard)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"[{model_name}] New best model saved with Val Loss: {val_loss:.6f}"
            )

    # Finalize SWA
    final_model_path = best_model_path
    if Config.USE_SWA:
        logger.info(f"[{model_name}] Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), swa_model_path)
        final_model_path = (
            swa_model_path  # Use SWA as the final model for this architecture
        )
        logger.info(f"[{model_name}] SWA model saved.")

    return final_model_path


def predict_with_tta(model, loader, device=Config.DEVICE):
    """
    Generates logits using Test-Time Augmentation (Horizontal Flip).
    Returns averaged logits.
    """
    model.eval()
    all_logits = []

    with torch.no_grad():
        for images, _ in loader:  # Test loader might return (img, id)
            images = images.to(device)

            # Forward pass original
            out_orig = model(images)

            # Forward pass flipped
            images_flip = torch.flip(images, dims=[3])
            out_flip = model(images_flip)

            # Average logits
            avg_logits = (out_orig + out_flip) / 2.0
            all_logits.append(avg_logits.cpu())

    return torch.cat(all_logits, dim=0)


def get_labels(loader):
    """Helper to extract all labels from a loader."""
    all_labels = []
    for _, labels in loader:
        all_labels.append(labels)
    return torch.cat(all_labels, dim=0)


def run_training():
    """
    Main orchestrator:
    1. Load Data
    2. Train Heterogeneous Ensemble
    3. Calibrate Models
    4. Generate Submission
    """
    logger.info("Starting Training Pipeline...")

    # 1. Load Data
    train_loader, val_loader, test_loader, class_list = get_loaders()

    ensemble_probs = []

    # 2. Iterate Architectures
    for model_name in Config.MODEL_ARCHS:
        logger.info(f"Processing Architecture: {model_name}")

        # Train
        saved_model_path = fit_model(model_name, train_loader, val_loader)

        # Load best/swa model for inference
        logger.info(
            f"Loading trained model from {saved_model_path} for inference/calibration."
        )
        model = create_model(model_name, num_classes=len(class_list), pretrained=False)

        # Handle SWA vs Standard state dict keys
        state_dict = torch.load(saved_model_path, map_location=Config.DEVICE)
        # If SWA model, keys might be prefixed with 'module.' or 'n_averaged' exists
        # AveragedModel saves standard keys usually, but let's be safe
        model.load_state_dict(state_dict, strict=False)
        model.to(Config.DEVICE)
        model.eval()

        # 3. Calibration (Temperature Scaling)
        if Config.USE_TEMP_SCALING:
            logger.info(f"[{model_name}] Performing Temperature Calibration...")
            # Get Validation Logits (with TTA for consistency)
            val_logits = predict_with_tta(model, val_loader)
            val_labels = get_labels(val_loader)

            scaler = TemperatureScaler()
            scaler.fit(val_logits, val_labels)

            # Get Test Logits
            test_logits = predict_with_tta(model, test_loader)

            # Apply Calibration and Softmax
            probs = scaler.get_probabilities(test_logits)
        else:
            # No calibration
            test_logits = predict_with_tta(model, test_loader)
            probs = torch.softmax(test_logits, dim=1)

        ensemble_probs.append(probs.numpy())

    # 4. Ensemble Aggregation
    logger.info("Aggregating Ensemble Predictions...")
    avg_probs = np.mean(ensemble_probs, axis=0)

    # 5. Generate Submission
    logger.info("Generating Submission File...")

    # Get Test IDs
    test_ids = []
    for _, batch_ids in test_loader:
        test_ids.extend(batch_ids)

    # Create DataFrame
    df_sub = pd.DataFrame(avg_probs, columns=class_list)
    df_sub.insert(0, "id", test_ids)

    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)

    logger.info(f"Submission saved to {sub_path}")
    logger.info("Training Pipeline Completed Successfully.")
