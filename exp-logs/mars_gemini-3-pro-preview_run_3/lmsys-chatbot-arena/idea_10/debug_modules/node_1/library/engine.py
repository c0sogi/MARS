import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from library.config import Config
from library.utils import get_logger, compute_metrics, get_device

logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, epoch):
    """
    Trains the model for one epoch using Gradient Accumulation and Mixed Precision.
    """
    model.train()
    total_loss = 0.0
    dataset_size = 0

    accumulation_steps = Config.gradient_accumulation_steps
    optimizer.zero_grad()

    num_steps = len(loader)

    for step, batch in enumerate(loader):
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        batch_size = batch["input_ids_a"].size(0)

        # Mixed Precision Forward Pass
        with autocast(enabled=Config.fp16):
            outputs = model(batch)
            targets = batch["target"]

            # CrossEntropyLoss works with soft probabilities
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward Pass
        scaler.scale(loss).backward()

        # Optimizer Step (only after accumulation steps)
        if (step + 1) % accumulation_steps == 0 or (step + 1) == num_steps:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Update Scheduler
            if scheduler is not None:
                scheduler.step()

            # Reset gradients
            optimizer.zero_grad()

        # Track Loss (scale back up for logging)
        total_loss += loss.item() * accumulation_steps * batch_size
        dataset_size += batch_size

    avg_loss = total_loss / dataset_size
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    dataset_size = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            batch_size = batch["input_ids_a"].size(0)

            with autocast(enabled=Config.fp16):
                outputs = model(batch)

            targets = batch["target"]
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / dataset_size
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss

    return metrics


def predict(model, loader, device):
    """
    Generates predictions for the test set, optionally using Test-Time Augmentation (TTA).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            # 1. Standard Prediction
            with autocast(enabled=Config.fp16):
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)

            # 2. Test-Time Augmentation (TTA)
            if Config.tta:
                # Create swapped batch: A <-> B
                # Scalars: [P, Ra, Rb] -> [P, Rb, Ra]
                scalars_swapped = torch.stack(
                    [
                        batch["scalars"][:, 0],
                        batch["scalars"][:, 2],
                        batch["scalars"][:, 1],
                    ],
                    dim=1,
                )

                batch_swapped = {
                    "input_ids_a": batch["input_ids_b"],
                    "attention_mask_a": batch["attention_mask_b"],
                    "token_type_ids_a": batch["token_type_ids_b"],
                    "input_ids_b": batch["input_ids_a"],
                    "attention_mask_b": batch["attention_mask_a"],
                    "token_type_ids_b": batch["token_type_ids_a"],
                    "scalars": scalars_swapped,
                }

                with autocast(enabled=Config.fp16):
                    logits_swapped = model(batch_swapped)
                    probs_swapped = torch.softmax(logits_swapped, dim=1)

                # Restore swapped probabilities to original order
                # Original: [Win A, Win B, Tie]
                # Swapped Output: [Win B (Input 1), Win A (Input 2), Tie]
                # Mapping: Swapped[0] -> Original[1], Swapped[1] -> Original[0], Swapped[2] -> Original[2]
                probs_swapped_restored = torch.zeros_like(probs_swapped)
                probs_swapped_restored[:, 0] = probs_swapped[:, 1]  # Win A
                probs_swapped_restored[:, 1] = probs_swapped[:, 0]  # Win B
                probs_swapped_restored[:, 2] = probs_swapped[:, 2]  # Tie

                # Average predictions
                probs = (probs + probs_swapped_restored) / 2.0

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def run(model, train_loader, val_loader, test_loader):
    """
    Main execution function: Train, Validate, and Predict.
    """
    device = get_device()
    model.to(device)

    # Optimizer
    # We group parameters to apply weight decay correctly (exclude bias/LayerNorm)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate)

    # Scheduler
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // Config.gradient_accumulation_steps
    max_train_steps = Config.epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    scaler = GradScaler(enabled=Config.fp16)

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {Config.epochs} epochs on {device}...")

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )

        val_metrics = validate(model, val_loader, device)
        val_loss = val_metrics["loss"]

        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.15f}"  # Full precision as requested
        )

        # Checkpointing and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            logger.info(f"New best model found! Saving to {Config.model_path}")
            torch.save(model.state_dict(), Config.model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            logger.info("Early stopping triggered.")
            break

    # Inference
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.model_path, map_location=device))

    logger.info("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # Create Submission File
    # We need the IDs from the test file.
    # Since test_loader is sequential and deterministic, we can read the test CSV to get IDs.
    test_df = pd.read_csv(Config.test_path)
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    submission.to_csv(Config.submission_path, index=False)
    logger.info(f"Submission saved to {Config.submission_path}")
