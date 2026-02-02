import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import get_logger, probabilistic_f1


class MCSINModel(nn.Module):
    """
    Multi-Contrast Single-Instance Network (MC-SIN).

    Architecture:
        - Backbone: EfficientNetV2-Small (Pretrained on ImageNet)
        - Input: 3 Channels (Standard, CLAHE, Gamma-Corrected)
        - Pooling: Global Average Pooling (GAP)
        - Head: Single Linear Layer (Logits)

    Note: Metadata fusion is omitted as the provided data pipeline (library/data.py)
    does not supply tabular features during iteration.
    """

    def __init__(self, pretrained=True):
        super(MCSINModel, self).__init__()

        # Create EfficientNetV2-S backbone
        # in_chans=3 matches the Multi-Contrast input
        # num_classes=1 for binary classification (Cancer vs Normal)
        # global_pool='avg' enables Global Average Pooling
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            global_pool="avg",
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        # x shape: (Batch, 3, 640, 640)
        # Returns logits: (Batch, 1)
        return self.backbone(x)


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    """
    Training loop for a single epoch.
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
        # Explicitly disable AMP for loss to handle high pos_weight stability
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


def evaluate(model, loader, device, return_preds=False):
    """
    Evaluates the model on validation set using Probabilistic F1.
    Aggregates predictions by prediction_id (Max) before metric calculation.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, pred_ids in loader:
            images = images.to(device, non_blocking=True)

            # Inference
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
    # This matches the submission strategy and SIL logic
    df_agg = (
        df_res.groupby("prediction_id")
        .agg(
            {
                "prob": "max",
                "target": "max",  # Target should be same for same ID, max is safe
            }
        )
        .reset_index()
    )

    # Calculate pF1
    pf1 = probabilistic_f1(df_agg["target"].values, df_agg["prob"].values)

    if return_preds:
        return pf1, df_agg
    return pf1


def predict_and_submit(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves submission file.
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

    # Save
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} with {len(submission)} rows.")


def run_training(train_loader, val_loader, test_loader):
    """
    Main execution function for training and inference.
    """
    logger = get_logger("main")
    device = torch.device(Config.DEVICE)

    # Initialize Model
    logger.info(f"Initializing {Config.MODEL_NAME}...")
    model = MCSINModel(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Loss Function
    # Critical: pos_weight on device for BCEWithLogitsLoss
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    # Scaler for AMP
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Training Loop
    best_pf1 = 0.0
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_pf1 = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Val pF1: {val_pf1:.6f}")

        # Checkpointing & Early Stopping
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved! (pF1: {val_pf1:.6f})")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # Load Best Model for Inference
    if os.path.exists(Config.MODEL_SAVE_PATH):
        logger.info("Loading best model for inference...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning("No best model saved. Using current model.")

    # Generate Submission
    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
