import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from library.config import Config
from library.utils import AverageMeter, kl_divergence_score, get_logger, seed_everything
from library.data_loader import get_dataloaders
from library.model import DualScaleSpectrogramNet


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """
    Trains the model for one epoch using MixUp augmentation if enabled.
    """
    model.train()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move data to device
        x_eeg = inputs[0].to(device, non_blocking=True)
        x_spec = inputs[1].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # MixUp Augmentation
        if config.USE_MIXUP and np.random.random() < 0.5:
            lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
            index = torch.randperm(x_eeg.size(0)).to(device)

            x_eeg = lam * x_eeg + (1 - lam) * x_eeg[index]
            x_spec = lam * x_spec + (1 - lam) * x_spec[index]
            targets_a, targets_b = targets, targets[index]

            # Forward pass
            outputs = model((x_eeg, x_spec))

            # Loss calculation (KLDivLoss expects LogProbs)
            log_outputs = torch.log(outputs + 1e-15)
            loss = lam * criterion(log_outputs, targets_a) + (1 - lam) * criterion(
                log_outputs, targets_b
            )
        else:
            # Standard Forward pass
            outputs = model((x_eeg, x_spec))
            log_outputs = torch.log(outputs + 1e-15)
            loss = criterion(log_outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), x_eeg.size(0))

        # Calculate KL on original targets for monitoring
        with torch.no_grad():
            kl = kl_divergence_score(targets, outputs)
            kl_meter.update(kl, x_eeg.size(0))

    return loss_meter.avg, kl_meter.avg


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    with torch.no_grad():
        for inputs, targets in loader:
            x_eeg = inputs[0].to(device, non_blocking=True)
            x_spec = inputs[1].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model((x_eeg, x_spec))

            log_outputs = torch.log(outputs + 1e-15)
            loss = criterion(log_outputs, targets)

            loss_meter.update(loss.item(), x_eeg.size(0))
            kl = kl_divergence_score(targets, outputs)
            kl_meter.update(kl, x_eeg.size(0))

    return loss_meter.avg, kl_meter.avg


def generate_submission(model, test_loader, test_df, config, logger=None):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    if logger is None:
        logger = get_logger(os.path.join(config.WORKING_DIR, "inference.log"))

    device = config.DEVICE
    model.to(device)

    # Load best model weights
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        logger.info(f"Loaded model from {config.MODEL_PATH} for inference.")
    else:
        logger.warning("Model path not found! Using current model weights.")

    model.eval()
    preds = []

    logger.info("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            x_eeg = inputs[0].to(device)
            x_spec = inputs[1].to(device)

            outputs = model((x_eeg, x_spec))
            preds.append(outputs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Prepare Submission DataFrame
    sub_cols = [c.replace("_prob", "_vote") for c in config.TARGET_COLS]

    sub_df = pd.DataFrame(preds, columns=sub_cols)
    sub_df["eeg_id"] = test_df["eeg_id"].values

    # Reorder to match submission format
    cols = ["eeg_id"] + sub_cols
    sub_df = sub_df[cols]

    # Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_FILE}")


def run_training(config=Config):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # Setup
    seed_everything(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    logger = get_logger(os.path.join(config.WORKING_DIR, "training.log"))
    device = config.DEVICE

    logger.info(f"Starting training on device: {device}")

    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Debug Mode
    if config.DEBUG:
        train_df = train_df.iloc[:200]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]
        logger.info("Debug mode: Reduced dataset size.")

    # Get DataLoaders (Handles caching internally)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # Initialize Model
    model = DualScaleSpectrogramNet(config)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # Loss Function (KL Divergence)
    criterion = nn.KLDivLoss(reduction="batchmean")

    # Training Loop
    best_kl = float("inf")
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS}")

        train_loss, train_kl = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        val_loss, val_kl = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        logger.info(f"  Train Loss: {train_loss} | Train KL: {train_kl}")
        logger.info(f"  Val Loss:   {val_loss} | Val KL:   {val_kl}")

        # Checkpointing
        if val_kl < best_kl:
            best_kl = val_kl
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            logger.info(f"  New Best Model Saved! KL: {best_kl}")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"Training Complete. Best Val KL: {best_kl}")

    # Generate Submission
    generate_submission(model, test_loader, test_df, config, logger)
