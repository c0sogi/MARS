import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    set_seed,
    AverageMeter,
    save_checkpoint,
    get_weighted_log_loss_score,
)
from library.loss import CervicalSpineLoss
from library.data import get_dataloaders
from library.model import CervicalFractureNet


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()

    losses = AverageMeter("Loss")
    loss_study_meter = AverageMeter("Loss Study")
    loss_slice_meter = AverageMeter("Loss Slice")
    loss_spatial_meter = AverageMeter("Loss Spatial")
    loss_anatomy_meter = AverageMeter("Loss Anatomy")

    optimizer.zero_grad()

    start_time = time.time()

    for step, batch in enumerate(train_loader):
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)

        # Prepare targets dict
        targets = {
            "study_labels": batch["study_labels"].to(device, non_blocking=True),
            "slice_fracture_labels": batch["slice_fracture_labels"].to(
                device, non_blocking=True
            ),
            "spatial_masks": batch["spatial_masks"].to(device, non_blocking=True),
            "anatomy_labels": batch["anatomy_labels"].to(device, non_blocking=True),
            "has_bbox": batch["has_bbox"].to(device, non_blocking=True),
            "has_segmentation": batch["has_segmentation"].to(device, non_blocking=True),
        }

        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            predictions = model(images)
            loss, metrics = criterion(predictions, targets)

            # Normalize loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass
        scaler.scale(loss).backward()

        # Update weights every ACCUMULATION_STEPS
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update meters (multiply by accumulation steps to get back original scale for logging)
        losses.update(loss.item() * Config.ACCUMULATION_STEPS, batch_size)
        loss_study_meter.update(metrics["loss_study"], batch_size)
        loss_slice_meter.update(metrics["loss_slice"], batch_size)
        loss_spatial_meter.update(metrics["loss_spatial"], batch_size)
        loss_anatomy_meter.update(metrics["loss_anatomy"], batch_size)

    elapsed = time.time() - start_time

    print(f"Epoch {epoch} Train Summary:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Total Loss: {losses.avg}")
    print(f"  Study Loss: {loss_study_meter.avg}")
    print(f"  Slice Loss: {loss_slice_meter.avg}")
    print(f"  Spatial Loss: {loss_spatial_meter.avg}")
    print(f"  Anatomy Loss: {loss_anatomy_meter.avg}")

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Runs validation on the validation set.
    """
    model.eval()

    losses = AverageMeter("Loss")

    # Store predictions and targets for metric calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)

            targets = {
                "study_labels": batch["study_labels"].to(device, non_blocking=True),
                "slice_fracture_labels": batch["slice_fracture_labels"].to(
                    device, non_blocking=True
                ),
                "spatial_masks": batch["spatial_masks"].to(device, non_blocking=True),
                "anatomy_labels": batch["anatomy_labels"].to(device, non_blocking=True),
                "has_bbox": batch["has_bbox"].to(device, non_blocking=True),
                "has_segmentation": batch["has_segmentation"].to(
                    device, non_blocking=True
                ),
            }

            batch_size = images.size(0)

            # Forward pass (no autocast needed for eval usually, but consistent with train)
            # Using standard float32 for validation stability
            predictions = model(images)
            loss, _ = criterion(predictions, targets)

            losses.update(loss.item(), batch_size)

            # Collect study logits for metric
            logits = predictions["study_logits"]
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets["study_labels"].cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Competition Metric
    # Weights: C1-C7 (1.0), Patient Overall (7.0)
    # Assuming columns are ordered: C1..C7, Patient_Overall
    comp_metric = get_weighted_log_loss_score(all_targets, all_preds)

    print(f"Validation Summary:")
    print(f"  Total Loss: {losses.avg}")
    print(f"  Weighted Log Loss (Metric): {comp_metric}")

    return losses.avg, comp_metric


def run_training():
    """
    Main execution function for training the model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = CervicalFractureNet()
    model.to(device)

    # 4. Loss & Optimizer
    criterion = CervicalSpineLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Mixed Precision Scaler
    scaler = GradScaler()

    # 5. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{Config.EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_metric = validate(val_loader, model, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Checkpointing & Early Stopping
        is_best = val_metric < best_metric
        if is_best:
            print(f"New Best Metric: {val_metric} (Previous: {best_metric})")
            best_metric = val_metric
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Save Checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
            },
            is_best,
            Config.WORKING_DIR,
        )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Weighted Log Loss: {best_metric}")
