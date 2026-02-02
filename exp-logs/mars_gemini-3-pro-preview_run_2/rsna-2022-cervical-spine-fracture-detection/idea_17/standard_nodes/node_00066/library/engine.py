import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import AverageMeter, get_weighted_log_loss, setup_logger


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    """
    Trains the model for one epoch using mixed precision and gradient accumulation.
    """
    model.train()
    loss_meter = AverageMeter()

    # Zero gradients at the start
    optimizer.zero_grad()

    num_steps = len(loader)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed Precision Context
        with autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = get_weighted_log_loss(logits, targets)

            # Normalize loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS

        # Scale loss and backward
        scaler.scale(loss).backward()

        # Update weights after accumulation steps
        if (step + 1) % Config.ACCUMULATION_STEPS == 0 or (step + 1) == num_steps:
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next accumulation cycle
            optimizer.zero_grad()

        # Record loss (multiply back by accumulation steps for logging)
        loss_meter.update(loss.item() * Config.ACCUMULATION_STEPS, images.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=(device == "cuda")):
                logits = model(images)
                loss = get_weighted_log_loss(logits, targets)

            loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def fit(model, train_loader, val_loader, device):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    logger = setup_logger()
    logger.info(f"Starting training on device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Gradient Scaler for AMP
    scaler = GradScaler(enabled=(device == "cuda"))

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Log Metrics (Full Precision)
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"LR: {curr_lr:.8f} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss: {val_loss:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < (best_loss - Config.MIN_DELTA):
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved with loss: {best_loss:.10f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    return best_loss


def predict_and_submit(model, test_loader, test_df, device):
    """
    Generates predictions for the test set and creates the submission file.
    """
    logger = setup_logger()
    logger.info("Starting inference...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Loaded best model checkpoint.")
    else:
        logger.warning("No checkpoint found. Using current model state.")

    model.eval()
    all_probs = []

    # Inference Loop
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)

            with autocast(enabled=(device == "cuda")):
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    # Concatenate all predictions (N_studies, 8)
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        # Handle empty test set case
        all_probs = np.zeros((0, 8))

    # Map predictions to StudyInstanceUIDs
    # Note: test_loader is not shuffled, so order matches test_df
    if len(test_df) != len(all_probs):
        logger.error(
            f"Mismatch: {len(test_df)} studies in metadata vs {len(all_probs)} predictions."
        )

    # Create a mapping: StudyUID -> {target_col: prob}
    study_preds = {}
    target_cols = Config.TARGET_COLS  # ["C1", ..., "patient_overall"]

    for idx, row in test_df.iterrows():
        uid = row["StudyInstanceUID"]
        if idx < len(all_probs):
            probs = all_probs[idx]
            study_preds[uid] = {col: p for col, p in zip(target_cols, probs)}

    # Generate Submission Rows
    # We iterate over the sample submission to ensure correct order and row_ids
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    submission_rows = []

    for _, row in sample_sub.iterrows():
        row_id = row["row_id"]
        # row_id format: "StudyInstanceUID_Target"
        # Example: "1.2.826.0.1.3680043.10001_C1"

        # Find the last underscore to split UID and Target
        split_idx = row_id.rfind("_")
        if split_idx == -1:
            submission_rows.append(0.0)
            continue

        study_uid = row_id[:split_idx]
        target = row_id[split_idx + 1 :]

        prob = 0.5  # Default
        if study_uid in study_preds and target in study_preds[study_uid]:
            prob = study_preds[study_uid][target]

        submission_rows.append(prob)

    # Save Submission
    sample_sub["fractured"] = submission_rows
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
