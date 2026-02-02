import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    probabilistic_f1,
    get_device,
    setup_logger,
    set_seed,
)
from library.data import get_dataloaders
from library.model import DSGEHNModel


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler):
    """
    Trains the model for one epoch using Deep Supervision and FP32-guarded loss.
    """
    model.train()
    losses = AverageMeter("Loss", ":.4f")

    # Iterate over batches (no progress bar as per requirements)
    for batch in loader:
        # Move data to device
        imgs = batch["image"].to(device, non_blocking=True)
        cats = batch["categorical"].to(device, non_blocking=True)
        conts = batch["continuous"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        # Forward Pass
        # Model returns (final_logits, aux_logits)
        final_logits, aux_logits = model(imgs, cats, conts)

        # Loss Calculation
        # Explicitly disable autocast for loss calculation to ensure FP32 precision.
        # This prevents NaN divergence when using high pos_weight.
        with (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if device.type == "cuda"
            else torch.no_grad()
        ):
            final_logits_fp32 = final_logits.float()
            aux_logits_fp32 = aux_logits.float()
            labels_fp32 = labels.float()

            loss_main = criterion(final_logits_fp32, labels_fp32)
            loss_aux = criterion(aux_logits_fp32, labels_fp32)

            # Weighted sum: L_total = L_final + 0.4 * L_aux
            total_loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        # Backward Pass
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Optimization Step
        optimizer.step()

        # Scheduler Step (OneCycleLR steps every batch)
        if scheduler is not None:
            scheduler.step()

        losses.update(total_loss.item(), imgs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter("Loss", ":.4f")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            cats = batch["categorical"].to(device, non_blocking=True)
            conts = batch["continuous"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            # Forward Pass (Main head only for validation)
            final_logits, _ = model(imgs, cats, conts)

            # Loss
            loss = criterion(final_logits, labels)
            losses.update(loss.item(), imgs.size(0))

            # Predictions for Metrics
            probs = torch.sigmoid(final_logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets)

    # Compute Probabilistic F1
    # flatten arrays to ensure correct shape
    all_targets = np.array(all_targets).flatten()
    all_preds = np.array(all_preds).flatten()

    pf1 = probabilistic_f1(all_targets, all_preds)

    return losses.avg, pf1


def run_training():
    """
    Main training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    log_file = os.path.join(Config.WORK_DIR, "train.log")
    logger = setup_logger(log_file)
    device = get_device()

    logger.info(f"Starting training on device: {device}")

    # 2. Data
    # Load cached data if available, otherwise process
    train_loader, val_loader, _, feature_meta = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model
    model = DSGEHNModel(feature_meta, pretrained=Config.PRETRAINED)
    model.to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    # 5. Loss Function
    # Handle class imbalance with positive weight
    pos_weight = torch.tensor(Config.POS_WEIGHT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 6. Training Loop
    best_pf1 = 0.0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scheduler
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Log metrics (Full precision)
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val pF1: {val_pf1}"
        )

        # Save Checkpoint
        is_best = val_pf1 > best_pf1
        if is_best:
            best_pf1 = val_pf1
            logger.info(f"New Best pF1: {best_pf1}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_pf1": best_pf1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=is_best,
        )

    logger.info("Training Complete.")
