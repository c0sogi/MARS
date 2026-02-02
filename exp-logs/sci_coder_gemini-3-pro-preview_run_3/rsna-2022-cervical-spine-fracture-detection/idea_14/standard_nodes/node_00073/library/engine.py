import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, get_logger


def train_one_epoch(loader, model, criterion, optimizer, device, epoch):
    """
    Performs one epoch of training using Automatic Mixed Precision (AMP).
    """
    model.train()
    losses = AverageMeter()

    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler("cuda")

    # Iterate over the loader (silent, no progress bar)
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def inference(loader, model, device, test_df):
    """
    Generates predictions for the test set and creates the submission file.
    """
    model.eval()
    preds = []

    # Generate probabilities
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                probs = torch.sigmoid(logits)
            preds.append(probs.float().cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Map predictions to submission format
    # Model output columns: 0-6 (C1-C7), 7 (patient_overall)
    col_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    submission_rows = []

    # test_df should correspond 1:1 with the loader batches (shuffle=False)
    study_uids = test_df["StudyInstanceUID"].values

    for i, study_uid in enumerate(study_uids):
        study_preds = preds[i]

        for class_idx, class_name in enumerate(col_names):
            row_id = f"{study_uid}_{class_name}"
            prob = study_preds[class_idx]
            submission_rows.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def fit(model, train_loader, val_loader, test_loader, test_df, criterion, device):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    logger = get_logger("training")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    # T_max is set based on total epochs * multiplier
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(Config.EPOCHS * Config.T_MAX_MULTIPLIER),
        eta_min=Config.MIN_LR,
    )

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(val_loader, model, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.16f}")
        logger.info(f"Val Loss:   {val_loss:.16f}")
        logger.info(f"LR:         {current_lr:.8f}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"Model saved (Best Val Loss: {best_val_loss:.16f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # Load best model for inference
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Inference
    if test_loader is not None and test_df is not None:
        logger.info("Generating submission...")
        inference(test_loader, model, device, test_df)
