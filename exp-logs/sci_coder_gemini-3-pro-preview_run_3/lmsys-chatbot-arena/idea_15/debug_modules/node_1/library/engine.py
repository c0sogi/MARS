import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, compute_log_loss, seed_everything
from library.data_processing import get_dataloaders
from library.model_components import SiameseDeberta


def train_fn(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Training loop for one epoch.
    Handles Gradient Accumulation and Mixed Precision Training.
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_FP16)

    accum_steps = Config.GRAD_ACCUM_STEPS

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device)
        mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        mask_b = batch["attention_mask_b"].to(device)
        scalars = batch["scalars"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids_a.size(0)

        # Forward pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
            logits = model(input_ids_a, mask_a, input_ids_b, mask_b, scalars)
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Scale loss for gradient accumulation
            if accum_steps > 1:
                loss = loss / accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Optimizer step (only every accum_steps)
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update metrics (scale loss back up for logging)
        loss_val = loss.item() * accum_steps if accum_steps > 1 else loss.item()
        losses.update(loss_val, batch_size)

    return losses.avg


def eval_fn(model, val_loader, device):
    """
    Validation loop.
    Computes Log Loss and Cross Entropy Loss.
    """
    model.eval()
    preds = []
    targets = []
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids_a.size(0)

            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                logits = model(input_ids_a, mask_a, input_ids_b, mask_b, scalars)
                loss = criterion(logits, labels)

            losses.update(loss.item(), batch_size)

            # Calculate probabilities
            probs = torch.softmax(logits, dim=1)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Compute metric
    log_loss_score = compute_log_loss(targets, preds)

    return log_loss_score, losses.avg


def inference_fn(model, test_loader, device):
    """
    Inference loop with Test-Time Augmentation (TTA).
    Predicts on (A, B) and (B, A), then averages the results.
    """
    model.eval()
    final_preds = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)
            batch_ids = batch["ids"]

            # --- Pass 1: Original Order (A, B) ---
            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                logits_1 = model(input_ids_a, mask_a, input_ids_b, mask_b, scalars)
                probs_1 = (
                    torch.softmax(logits_1, dim=1).cpu().numpy()
                )  # [Win_A, Win_B, Tie]

            # --- Pass 2: Swapped Order (B, A) ---
            # Swap inputs
            # Scalars: [p_len, r_len_a, r_len_b] -> [p_len, r_len_b, r_len_a]
            scalars_swapped = scalars.clone()
            scalars_swapped[:, 1] = scalars[:, 2]
            scalars_swapped[:, 2] = scalars[:, 1]

            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                # Note: input_ids_b goes into branch A slot, input_ids_a goes into branch B slot
                logits_2 = model(
                    input_ids_b, mask_b, input_ids_a, mask_a, scalars_swapped
                )
                probs_2 = (
                    torch.softmax(logits_2, dim=1).cpu().numpy()
                )  # [Win_B_swapped, Win_A_swapped, Tie]

            # Realign probs_2 to match [Win_A, Win_B, Tie] perspective
            # probs_2[0] is probability that the FIRST input (which is B) wins -> Win_B
            # probs_2[1] is probability that the SECOND input (which is A) wins -> Win_A
            # probs_2[2] is Tie

            probs_2_aligned = np.zeros_like(probs_2)
            probs_2_aligned[:, 0] = probs_2[:, 1]  # Win_A
            probs_2_aligned[:, 1] = probs_2[:, 0]  # Win_B
            probs_2_aligned[:, 2] = probs_2[:, 2]  # Tie

            # Average predictions
            avg_probs = (probs_1 + probs_2_aligned) / 2.0

            final_preds.append(avg_probs)
            ids.extend(batch_ids)

    final_preds = np.concatenate(final_preds)
    return ids, final_preds


def run():
    """
    Main execution function.
    Sets up data, model, runs training with early stopping, and generates submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        debug=Config.DEBUG,
        batch_size=Config.TRAIN_BATCH_SIZE,
        val_batch_size=Config.VALID_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = SiameseDeberta()
    model.to(device)

    # 3. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate steps for scheduler
    num_training_steps = (len(train_loader) // Config.GRAD_ACCUM_STEPS) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)

        # Validate
        val_log_loss, val_ce_loss = eval_fn(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Log Loss: {val_log_loss}")
        print(f"Val CE Loss: {val_ce_loss}")

        # Early Stopping & Model Checkpointing
        if val_log_loss < best_loss:
            best_loss = val_log_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference & Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    ids, predictions = inference_fn(model, test_loader, device)

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
