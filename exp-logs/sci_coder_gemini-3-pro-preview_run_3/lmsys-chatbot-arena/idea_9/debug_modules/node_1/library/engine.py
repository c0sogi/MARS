import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import get_logger, compute_log_loss
from library.model import SiameseDebertaModel

logger = get_logger("engine")


def train_one_epoch(model, optimizer, scheduler, dataloader, device, scaler, epoch):
    """
    Trains the model for one epoch using Gradient Accumulation and Mixed Precision.
    """
    model.train()

    dataset_size = len(dataloader.dataset)
    running_loss = 0.0
    optimizer.zero_grad()

    num_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        # Mixed Precision Forward Pass
        with autocast():
            logits, loss = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                features=features,
                target=targets,
            )

            # Normalize loss for gradient accumulation
            loss = loss / Config.gradient_accumulation_steps

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Accumulate Loss for logging
        running_loss += loss.item() * Config.gradient_accumulation_steps

        # Optimizer Step (only when accumulation is complete)
        if (step + 1) % Config.gradient_accumulation_steps == 0 or (
            step + 1
        ) == num_steps:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Step optimizer and scheduler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if (step + 1) % 100 == 0:
            avg_loss = running_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1} | Step {step+1}/{num_steps} | Loss: {avg_loss:.6f} | LR: {lr:.2e}"
            )

    epoch_loss = running_loss / num_steps
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    all_targets = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            features = batch["features"].to(device)
            targets = batch["target"].to(device)

            # Forward pass (no autocast needed for validation usually, but safe to keep consistent)
            with autocast():
                logits, loss = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    features=features,
                    target=targets,
                )

            running_loss += loss.item()

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().float().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute Metric
    val_log_loss = compute_log_loss(all_targets, all_preds)
    avg_loss = running_loss / len(dataloader)

    return avg_loss, val_log_loss


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Averages predictions from (A, B) and (B, A).
    """
    model.eval()
    all_preds = []

    logger.info("Starting TTA Prediction...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # --- Pass 1: Original (A, B) ---
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            features = batch["features"].to(device)  # [log_p, log_a, log_b]

            with autocast():
                logits_orig, _ = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    features=features,
                )
            probs_orig = torch.softmax(logits_orig, dim=1)  # [A_win, B_win, Tie]

            # --- Pass 2: Swapped (B, A) ---
            # Swap input tensors
            input_ids_a_swap = input_ids_b
            attention_mask_a_swap = attention_mask_b
            input_ids_b_swap = input_ids_a
            attention_mask_b_swap = attention_mask_a

            # Swap scalar features: [log_p, log_a, log_b] -> [log_p, log_b, log_a]
            # features is shape (batch, 3)
            features_swap = torch.stack(
                [features[:, 0], features[:, 2], features[:, 1]], dim=1
            )

            with autocast():
                logits_swap, _ = model(
                    input_ids_a=input_ids_a_swap,
                    attention_mask_a=attention_mask_a_swap,
                    input_ids_b=input_ids_b_swap,
                    attention_mask_b=attention_mask_b_swap,
                    features=features_swap,
                )
            probs_swap = torch.softmax(logits_swap, dim=1)
            # probs_swap output is [Winner=SwapA, Winner=SwapB, Tie]
            # Since SwapA is OriginalB and SwapB is OriginalA:
            # probs_swap is effectively [Winner=OriginalB, Winner=OriginalA, Tie]

            # Map back to original order: [Winner=OriginalA, Winner=OriginalB, Tie]
            probs_swap_mapped = torch.stack(
                [
                    probs_swap[:, 1],  # Original A (was Swap B)
                    probs_swap[:, 0],  # Original B (was Swap A)
                    probs_swap[:, 2],  # Tie
                ],
                dim=1,
            )

            # --- Average ---
            avg_probs = (probs_orig + probs_swap_mapped) / 2.0
            all_preds.append(avg_probs.cpu().float().numpy())

            if (i + 1) % 50 == 0:
                logger.info(f"Predicted batch {i+1}/{len(dataloader)}")

    return np.concatenate(all_preds, axis=0)


def run_training(train_loader, val_loader, test_loader):
    """
    Main function to run the training pipeline, validation, and submission generation.
    """
    device = Config.device
    logger.info(f"Using device: {device}")

    # 1. Initialize Model
    logger.info("Initializing Model...")
    model = SiameseDebertaModel()
    model.to(device)

    # 2. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
        eps=Config.eps,
    )

    num_training_steps = (
        len(train_loader) * Config.epochs // Config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler(enabled=Config.use_fp16)

    # 3. Training Loop
    best_val_loss = float("inf")

    logger.info("Starting Training...")
    for epoch in range(Config.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{Config.epochs} ===")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, scaler, epoch
        )
        logger.info(f"Epoch {epoch + 1} Training Loss: {train_loss:.6f}")

        # Validate
        val_loss_ce, val_log_loss = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch + 1} Validation CE Loss: {val_loss_ce:.6f}")
        logger.info(f"Epoch {epoch + 1} Validation Log Loss: {val_log_loss}")

        # Save Best Model
        if val_log_loss < best_val_loss:
            logger.info(
                f"Validation Metric Improved ({best_val_loss} -> {val_log_loss}). Saving model..."
            )
            best_val_loss = val_log_loss
            torch.save(model.state_dict(), Config.model_save_path)
        else:
            logger.info(f"Validation Metric did not improve (Best: {best_val_loss}).")

    # 4. Inference on Test Set
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    predictions = predict_with_tta(model, test_loader, device)

    # 5. Create Submission
    logger.info("Generating submission file...")

    # Load test IDs from metadata
    test_df = pd.read_csv(Config.test_path)
    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    submission_df.to_csv(Config.submission_path, index=False)
    logger.info(f"Submission saved to {Config.submission_path}")
    logger.info("Done.")
