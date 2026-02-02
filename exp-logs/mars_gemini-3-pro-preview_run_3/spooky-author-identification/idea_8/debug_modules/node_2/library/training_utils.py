import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from library.config import TrainConfig, PathConfig, ModelConfig
from library.utils import calculate_log_loss
from library.models import AWP


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    awp=None,
    scaler=None,
):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        epoch (int): Current epoch index (1-based for AWP logic).
        awp (AWP, optional): Adversarial Weight Perturbation object.
        scaler (GradScaler, optional): For mixed precision training.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move batch to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "aux_targets"}
        # aux_targets might be needed if MTL is on
        if "aux_targets" in batch:
            inputs["aux_targets"] = batch["aux_targets"].to(device)

        # Forward pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(**inputs)
            loss = outputs["loss"]

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()

            if awp is not None:
                # AWP attack requires a second forward/backward pass logic
                # The provided AWP class handles the logic internally via attack_backward
                # which does: save -> attack -> forward -> backward -> restore
                awp.attack_backward(inputs, epoch)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TrainConfig.MAX_GRAD_NORM
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if awp is not None:
                awp.attack_backward(inputs, epoch)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TrainConfig.MAX_GRAD_NORM
            )
            optimizer.step()

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to validate on.

    Returns:
        tuple: (val_loss, predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    # We don't track the model's internal loss during validation for the metric,
    # we calculate the competition metric (Log Loss) on the probabilities.

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "aux_targets"}
            # Labels are needed for metric calculation later
            labels = batch["labels"].numpy()

            # Forward pass
            # We use autocast for inference speedup on A100
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
                logits = outputs["logits"]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate Competition Metric
    val_loss = calculate_log_loss(all_labels, all_preds)

    return val_loss, all_preds, all_labels


def run_fold_training(
    model, train_loader, val_loader, fold_idx, backbone_name, epochs=TrainConfig.EPOCHS
):
    """
    Orchestrates the training for a single fold, including optimization,
    scheduling, AWP, and early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        fold_idx (int): Current fold index.
        backbone_name (str): Name of the backbone (for saving).
        epochs (int): Number of epochs to train.

    Returns:
        tuple: (best_val_loss, best_predictions)
    """
    device = TrainConfig.DEVICE
    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=TrainConfig.LR, weight_decay=TrainConfig.WEIGHT_DECAY
    )

    # Scheduler
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * TrainConfig.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    # AWP
    awp = None
    if TrainConfig.USE_AWP:
        print(f"Initializing AWP (Start Epoch: {TrainConfig.AWP_START_EPOCH})...")
        awp = AWP(
            model,
            optimizer,
            adv_lr=TrainConfig.AWP_LR,
            adv_eps=TrainConfig.AWP_EPS,
            start_epoch=TrainConfig.AWP_START_EPOCH,
            scaler=scaler,
        )

    # Early Stopping & Checkpointing
    best_val_loss = float("inf")
    best_preds = None
    patience_counter = 0
    safe_backbone = backbone_name.replace("/", "-")
    save_path = os.path.join(
        PathConfig.FINETUNED_MODELS_DIR,
        f"best_model_{safe_backbone}_fold_{fold_idx}.pt",
    )

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, awp, scaler
        )

        # Validate
        val_loss, preds, _ = validate_one_epoch(model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = preds
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Model Saved! Loss: {best_val_loss}")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{TrainConfig.PATIENCE}")

        if patience_counter >= TrainConfig.PATIENCE:
            print("  >>> Early Stopping Triggered.")
            break

    # Load best model weights before returning
    # This ensures the model object passed in reflects the best state
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return best_val_loss, best_preds
