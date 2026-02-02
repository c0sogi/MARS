import os
import gc
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, get_logger
from library.model import DualStreamSiameseModel

logger = get_logger("engine", "engine.log")


def train_fn(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    """
    Training loop for a single epoch.
    """
    model.train()

    # Enable gradient checkpointing if configured and available
    if Config.GRADIENT_CHECKPOINTING:
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    loss_meter = AverageMeter()

    # CrossEntropyLoss supports soft probability targets
    loss_fn = nn.CrossEntropyLoss()

    count = 0

    for batch in dataloader:
        # Move inputs to device
        ids_a = batch["input_ids_a"].to(device)
        mask_a = batch["attention_mask_a"].to(device)
        type_a = batch["token_type_ids_a"].to(device)

        ids_b = batch["input_ids_b"].to(device)
        mask_b = batch["attention_mask_b"].to(device)
        type_b = batch["token_type_ids_b"].to(device)

        scalars = batch["scalars"].to(device)
        targets = batch["target"].to(device)

        batch_size = ids_a.size(0)

        # Mixed Precision Forward Pass
        with autocast(enabled=Config.USE_FP16):
            logits = model(
                input_ids_a=ids_a,
                attention_mask_a=mask_a,
                token_type_ids_a=type_a,
                input_ids_b=ids_b,
                attention_mask_b=mask_b,
                token_type_ids_b=type_b,
                scalars=scalars,
            )
            loss = loss_fn(logits, targets)

        # Normalize loss for gradient accumulation
        if Config.GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        # Backward Pass
        scaler.scale(loss).backward()

        # Optimizer Step
        if (count + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        loss_meter.update(loss.item() * Config.GRADIENT_ACCUMULATION_STEPS, batch_size)
        count += 1

    return loss_meter.avg


def eval_fn(model, dataloader, device):
    """
    Validation loop.
    """
    model.eval()
    loss_meter = AverageMeter()
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            type_a = batch["token_type_ids_a"].to(device)

            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            type_b = batch["token_type_ids_b"].to(device)

            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            batch_size = ids_a.size(0)

            with autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a=ids_a,
                    attention_mask_a=mask_a,
                    token_type_ids_a=type_a,
                    input_ids_b=ids_b,
                    attention_mask_b=mask_b,
                    token_type_ids_b=type_b,
                    scalars=scalars,
                )
                loss = loss_fn(logits, targets)

            loss_meter.update(loss.item(), batch_size)

    return loss_meter.avg


def inference_fn(model, dataloader, device):
    """
    Inference loop with Test-Time Augmentation (TTA).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            type_a = batch["token_type_ids_a"].to(device)

            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            type_b = batch["token_type_ids_b"].to(device)

            scalars = batch["scalars"].to(device)

            # 1. Standard Forward Pass
            with autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a=ids_a,
                    attention_mask_a=mask_a,
                    token_type_ids_a=type_a,
                    input_ids_b=ids_b,
                    attention_mask_b=mask_b,
                    token_type_ids_b=type_b,
                    scalars=scalars,
                )

            probs = torch.softmax(logits, dim=1)

            # 2. Test-Time Augmentation (Swap A and B)
            if Config.TTA:
                # Swap scalars: [p, a, b] -> [p, b, a]
                # indices: 0->0, 1->2, 2->1
                scalars_inv = scalars[:, [0, 2, 1]]

                with autocast(enabled=Config.USE_FP16):
                    logits_inv = model(
                        input_ids_a=ids_b,  # Swap A and B inputs
                        attention_mask_a=mask_b,
                        token_type_ids_a=type_b,
                        input_ids_b=ids_a,
                        attention_mask_b=mask_a,
                        token_type_ids_b=type_a,
                        scalars=scalars_inv,
                    )

                probs_inv = torch.softmax(logits_inv, dim=1)

                # Swap output probabilities back to align with original order
                # Current: [Win B_input, Win A_input, Tie]
                # Target:  [Win A_input, Win B_input, Tie]
                # Indices: 0->1, 1->0, 2->2
                probs_inv = probs_inv[:, [1, 0, 2]]

                # Average predictions
                probs = (probs + probs_inv) / 2.0

            preds.append(probs.cpu().numpy())

    return np.concatenate(preds)


def run_training(train_loader, val_loader):
    """
    Orchestrates the training process, including optimization and model saving.
    """
    device = Config.DEVICE

    # Initialize Model
    model = DualStreamSiameseModel()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_train_steps = int(
        len(train_loader) * Config.EPOCHS / Config.GRADIENT_ACCUMULATION_STEPS
    )
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Scaler for FP16
    scaler = GradScaler(enabled=Config.USE_FP16)

    best_loss = float("inf")

    logger.info(f"Starting training for {Config.EPOCHS} epochs on {device}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )

        # Validate
        val_loss = eval_fn(model, val_loader, device)

        elapsed = time.time() - start_time

        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {elapsed:.0f}s")
        logger.info(f"Train Loss: {train_loss}")
        # Print full precision as requested
        print(f"Epoch {epoch+1} Val Loss: {val_loss}")

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            logger.info(
                f"Validation Loss improved. Saving model to {Config.BEST_MODEL_PATH}"
            )
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    logger.info(f"Training complete. Best Val Loss: {best_loss}")

    # Clear memory
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()


def generate_submission(test_loader):
    """
    Loads the best model, runs inference on the test set, and saves the submission file.
    """
    device = Config.DEVICE

    logger.info("Loading best model for inference...")
    model = DualStreamSiameseModel()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)

    logger.info("Running inference on test set...")
    predictions = inference_fn(model, test_loader, device)

    # Load sample submission to get IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Ensure lengths match
    if len(predictions) != len(sample_sub):
        logger.error(
            f"Prediction length {len(predictions)} does not match sample submission {len(sample_sub)}"
        )

    # Assign probabilities
    sample_sub["winner_model_a"] = predictions[:, 0]
    sample_sub["winner_model_b"] = predictions[:, 1]
    sample_sub["winner_tie"] = predictions[:, 2]

    # Save
    logger.info(f"Saving submission to {Config.OUTPUT_SUBMISSION_PATH}")
    sample_sub.to_csv(Config.OUTPUT_SUBMISSION_PATH, index=False)

    return sample_sub
