import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_CLIP,
    SCHEDULER_T_0,
    SCHEDULER_T_MULT,
    MIN_LR,
    MAX_EPOCHS,
    PATIENCE,
    MODEL_SAVE_PATH,
    EARLY_STOP_METRIC,
)
from library.utils import MetricLogger


def train_one_epoch(model, loader, optimizer, criterion, device, standardizer):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move batch to device
        # The batch is a dictionary of tensors
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Get targets and types
        targets = batch["coupling_value"]
        types = batch["coupling_type"]

        # Standardize targets (Z-score normalization per type)
        # We predict in the standardized space to handle different scales
        norm_targets = standardizer.transform(targets, types)

        # Forward pass
        preds = model(batch)

        # Compute loss
        loss = criterion(preds, norm_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Update weights
        optimizer.step()

        # Accumulate statistics
        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def evaluate(model, loader, device, standardizer):
    """
    Evaluates the model on a validation/test set.
    Computes the competition metric (Log MAE).
    """
    model.eval()
    logger = MetricLogger()

    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass (predictions are in standardized space)
            preds = model(batch)

            # Get metadata for inverse transform
            types = batch["coupling_type"]
            targets = batch["coupling_value"]  # These are physical values

            # Inverse transform predictions to physical space
            phys_preds = standardizer.inverse_transform(preds, types)

            # Log for metric computation
            logger.update(phys_preds, targets, types)

    # Compute final metrics
    # score is Mean of Log(MAE) across types
    # type_scores is dict of MAE per type
    score, type_scores = logger.compute_metric()
    return score, type_scores


def train_model(
    model,
    train_loader,
    val_loader,
    standardizer,
    device=DEVICE,
    epochs=MAX_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    save_path=MODEL_SAVE_PATH,
):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    print(f"Starting training on device: {device}")

    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup Scheduler (Cosine Annealing Warm Restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T_0, T_mult=SCHEDULER_T_MULT, eta_min=MIN_LR
    )

    # Loss Function (L1 Loss / MAE)
    criterion = nn.L1Loss()

    # Early Stopping State
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Training ---
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, standardizer
        )

        # --- Validation ---
        val_score, val_type_maes = evaluate(model, val_loader, device, standardizer)

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Logging ---
        duration = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | Time: {duration:.2f}s | LR: {current_lr:.2e}")
        print(f"  Train Loss (Std MAE): {train_loss}")
        print(f"  Val LogMAE: {val_score}")

        # Detailed type metrics
        # print("  Val MAE per Type:")
        # for t, mae in val_type_maes.items():
        #     print(f"    Type {t}: {mae}")

        # --- Early Stopping & Checkpointing ---
        # The metric to minimize is LogMAE
        current_metric = val_score

        if current_metric < best_metric:
            print(
                f"  New best model! (Previous: {best_metric}, Current: {current_metric})"
            )
            best_metric = current_metric
            patience_counter = 0

            # Save Model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val LogMAE: {best_metric}")
    return best_metric
