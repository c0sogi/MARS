import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import get_logger, probabilistic_f1
from library.data import get_dataloaders
from library.model import MCSINModel


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    """
    Executes one epoch of training.

    Implements FP32-Guarded High-Weight Loss to prevent numerical instability
    when using high pos_weight with Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    logger = get_logger("train")
    start_time = time.time()

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Forward Pass
        with autocast(enabled=Config.USE_AMP):
            logits = model(images)

        # FP32-Guarded Loss Calculation
        # We explicitly disable autocast for the loss computation to ensure
        # the high pos_weight (20.0) doesn't cause overflow/NaNs in float16.
        with autocast(enabled=False):
            loss = criterion(logits.float(), targets.float())

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    elapsed = time.time() - start_time
    logger.info(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

    return avg_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions by prediction_id (Max) before calculating pF1.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, pred_ids in loader:
            images = images.to(device, non_blocking=True)

            with autocast(enabled=Config.USE_AMP):
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = targets.numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets)
            all_ids.extend(pred_ids)

    # Convert to DataFrame for aggregation
    df_res = pd.DataFrame(
        {"prediction_id": all_ids, "prob": all_preds, "target": all_targets}
    )

    # Aggregate by prediction_id: Take MAX probability per ID
    # This aligns with the Single-Instance Learning (SIL) strategy where
    # any suspicious view should trigger a high patient/side risk score.
    df_agg = (
        df_res.groupby("prediction_id")
        .agg({"prob": "max", "target": "max"})
        .reset_index()
    )

    # Calculate Probabilistic F1
    pf1 = probabilistic_f1(df_agg["target"].values, df_agg["prob"].values)

    return pf1


def predict_and_submit(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    all_preds = []
    all_ids = []
    logger = get_logger("inference")

    logger.info("Starting inference on test set...")

    with torch.no_grad():
        for images, _, pred_ids in loader:
            images = images.to(device, non_blocking=True)

            with autocast(enabled=Config.USE_AMP):
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_ids.extend(pred_ids)

    # Create DataFrame
    df_res = pd.DataFrame({"prediction_id": all_ids, "cancer": all_preds})

    # Aggregate: Max probability per prediction_id
    submission = df_res.groupby("prediction_id")["cancer"].max().reset_index()

    # Save submission
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} with {len(submission)} rows.")


def run(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Main execution function for the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    logger = get_logger("engine")
    device = torch.device(Config.DEVICE)

    # Update Config for this run if arguments differ
    Config.EPOCHS = epochs
    Config.DEBUG = debug

    # 1. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 2. Model Initialization
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = MCSINModel(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 3. Setup Training Components
    # Loss: BCEWithLogitsLoss with high pos_weight
    # Note: pos_weight must be on the same device as targets
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    scaler = GradScaler(enabled=Config.USE_AMP)

    # 4. Training Loop with Early Stopping
    best_pf1 = 0.0
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_pf1 = evaluate(model, val_loader, device)
        # Print full precision as requested
        logger.info(f"Epoch {epoch+1} | Val pF1: {val_pf1}")

        # Checkpointing
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved! (pF1: {val_pf1})")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # 5. Inference
    if os.path.exists(Config.MODEL_SAVE_PATH):
        logger.info("Loading best model for inference...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning("No best model saved. Using current model state.")

    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
