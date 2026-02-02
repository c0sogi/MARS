import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import get_logger, compute_log_loss
from library.model import SiameseDeberta

# Initialize logger
logger = get_logger("Engine")


def train_fn(model, dataloader, optimizer, scheduler, device, scaler):
    """
    Executes one training epoch.
    """
    model.train()
    final_loss = 0
    count = 0

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        scalars = batch["scalars"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=Config.FP16):
            outputs = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalars=scalars,
                labels=labels,
            )
            loss = outputs["loss"]

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        final_loss += loss.item()
        count += 1

        # Memory cleanup for safety
        del (
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            scalars,
            labels,
            outputs,
            loss,
        )

    return final_loss / count


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    preds = []
    targets = []
    final_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass (no autocast needed for eval usually, but consistent with train)
            outputs = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalars=scalars,
                labels=labels,
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            final_loss += loss.item()
            count += 1

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = final_loss / count
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate metric using the utility function
    metric_score = compute_log_loss(targets, preds)

    return avg_loss, metric_score


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping.
    """
    device = Config.DEVICE

    # Initialize Model
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = SiameseDeberta()
    model.to(device)

    # Optimizer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=Config.LEARNING_RATE)

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=Config.FP16)

    # Training Loop variables
    best_loss = np.inf
    patience_counter = 0

    logger.info(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, scaler)

        # Validate
        val_loss, val_metric = eval_fn(model, val_loader, device)

        logger.info(
            f"Epoch {epoch + 1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Loss: {val_loss:.8f} - "
            f"Val Metric (LogLoss): {val_metric:.16f}"
        )

        # Early Stopping and Model Saving
        if val_loss < best_loss:
            logger.info(
                f"Validation loss improved from {best_loss:.8f} to {val_loss:.8f}. Saving model..."
            )
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Training complete. Best Validation Loss: {best_loss:.8f}")


def predict_and_submit(test_loader):
    """
    Generates predictions using Test-Time Augmentation (TTA) and saves to submission.csv.
    """
    device = Config.DEVICE

    # Load Best Model
    logger.info("Loading best model for inference...")
    model = SiameseDeberta()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    final_preds = []

    logger.info("Starting inference with Test-Time Augmentation (TTA)...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)  # [len_p, len_a, len_b]

            # --- Pass 1: Original (A, B) ---
            outputs_1 = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalars=scalars,
            )
            probs_1 = torch.softmax(
                outputs_1["logits"], dim=1
            )  # [Batch, 3] -> [Win A, Win B, Tie]

            # --- Pass 2: Swapped (B, A) ---
            # Swap input IDs and masks
            # Swap scalar lengths: [len_p, len_a, len_b] -> [len_p, len_b, len_a]
            scalars_swapped = scalars.clone()
            scalars_swapped[:, 1] = scalars[:, 2]
            scalars_swapped[:, 2] = scalars[:, 1]

            outputs_2 = model(
                input_ids_a=input_ids_b,  # A becomes B
                attention_mask_a=attention_mask_b,
                input_ids_b=input_ids_a,  # B becomes A
                attention_mask_b=attention_mask_a,
                scalars=scalars_swapped,
            )
            probs_2 = torch.softmax(
                outputs_2["logits"], dim=1
            )  # [Batch, 3] -> [Win B (new A), Win A (new B), Tie]

            # --- Aggregate Predictions ---
            # probs_1: [P(A wins), P(B wins), P(Tie)]
            # probs_2: [P(B wins), P(A wins), P(Tie)] (relative to original A, B)

            # Average A wins: (probs_1[0] + probs_2[1]) / 2
            p_a = (probs_1[:, 0] + probs_2[:, 1]) / 2.0

            # Average B wins: (probs_1[1] + probs_2[0]) / 2
            p_b = (probs_1[:, 1] + probs_2[:, 0]) / 2.0

            # Average Tie: (probs_1[2] + probs_2[2]) / 2
            p_tie = (probs_1[:, 2] + probs_2[:, 2]) / 2.0

            batch_preds = torch.stack([p_a, p_b, p_tie], dim=1)
            final_preds.append(batch_preds.cpu().numpy())

    final_preds = np.concatenate(final_preds)

    # Create Submission DataFrame
    # We need the IDs. Since the loader preserves order and we process sequentially,
    # we can load the test csv or sample submission to get IDs.
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Safety check
    if len(final_preds) != len(sample_sub):
        logger.warning(
            f"Prediction count {len(final_preds)} does not match sample submission {len(sample_sub)}. Truncating or padding may occur."
        )
        # In this controlled environment, we assume they match if data processing is correct.
        # If debug mode was on, lengths differ.
        if Config.DEBUG:
            sample_sub = sample_sub.iloc[: len(final_preds)]

    submission = pd.DataFrame(
        {
            "id": sample_sub["id"],
            "winner_model_a": final_preds[:, 0],
            "winner_model_b": final_preds[:, 1],
            "winner_tie": final_preds[:, 2],
        }
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(submission.head().to_string())
