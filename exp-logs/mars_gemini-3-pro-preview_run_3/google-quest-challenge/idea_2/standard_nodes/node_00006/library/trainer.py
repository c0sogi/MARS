import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

from library.config import (
    DEVICE,
    MODEL_STATE_DICT_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS,
    seed_everything,
    TARGET_COLS,
    WORKING_DIR,
)
from library.dataset import get_dataloaders
from library.model import SiameseNetwork


def compute_spearman_metric(preds, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds (np.ndarray): Predictions of shape (N, num_targets).
        targets (np.ndarray): Ground truth labels of shape (N, num_targets).

    Returns:
        float: The mean Spearman correlation across all columns.
    """
    corrs = []
    # Iterate over each target column
    for col_idx in range(preds.shape[1]):
        p = preds[:, col_idx]
        t = targets[:, col_idx]

        # Check for constant values to avoid warnings/errors
        # Spearman correlation is undefined if one variable is constant
        if np.std(p) == 0 or np.std(t) == 0:
            corrs.append(np.nan)
        else:
            # spearmanr returns (correlation, pvalue) or a result object
            # We access the correlation coefficient (index 0)
            val = spearmanr(p, t)[0]
            corrs.append(val)

    return np.nanmean(corrs)


def train_stage_1(debug=False, load_cached_data=True):
    """
    Executes Stage 1: End-to-End Fine-Tuning of the Siamese Network.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
        load_cached_data (bool): If True, attempts to load pre-processed parquet files.

    Returns:
        float: The best validation Spearman correlation score achieved.
    """
    # 1. Setup
    seed_everything()
    print(f"Starting Stage 1 Training on device: {DEVICE}")

    # Ensure working directory for model exists
    os.makedirs(os.path.dirname(MODEL_STATE_DICT_PATH), exist_ok=True)

    # 2. Data Loading
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Model Initialization
    model = SiameseNetwork().to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Calculate total training steps
    # Note: We use integer division for steps per epoch
    steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
    num_training_steps = steps_per_epoch * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Loss Function
    # Targets are continuous probabilities [0,1], model outputs logits.
    # BCEWithLogitsLoss is appropriate for multi-label regression/classification in this range.
    criterion = nn.BCEWithLogitsLoss()

    # Initialize GradScaler for Mixed Precision
    scaler = GradScaler()

    # 6. Training Loop
    best_val_score = -1.0
    patience = 0
    patience_limit = 3  # Early stopping patience

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            q_input_ids = batch["q_input_ids"].to(DEVICE)
            q_attention_mask = batch["q_attention_mask"].to(DEVICE)
            a_input_ids = batch["a_input_ids"].to(DEVICE)
            a_attention_mask = batch["a_attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass with Autocast
            with autocast():
                logits = model(
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                )
                loss = criterion(logits, labels)
                # Normalize loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward pass with Scaler
            scaler.scale(loss).backward()

            # Accumulate loss for logging (scale back up)
            train_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # Optimizer Step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                train_steps += 1

        avg_train_loss = train_loss_sum / max(1, train_steps)
        print(f"Train Loss: {avg_train_loss}")

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                q_input_ids = batch["q_input_ids"].to(DEVICE)
                q_attention_mask = batch["q_attention_mask"].to(DEVICE)
                a_input_ids = batch["a_input_ids"].to(DEVICE)
                a_attention_mask = batch["a_attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with autocast():
                    logits = model(
                        q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                    )
                # Convert logits to probabilities for metric calculation
                probs = torch.sigmoid(logits)

                val_preds.append(probs.cpu().float().numpy())
                val_targets.append(labels.cpu().float().numpy())

        # Concatenate all batches
        if len(val_preds) > 0:
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)

            val_score = compute_spearman_metric(val_preds, val_targets)
        else:
            val_score = 0.0

        print(f"Validation Spearman Correlation: {val_score}")

        # --- Checkpointing & Early Stopping ---
        if val_score > best_val_score:
            print(
                f"Score improved from {best_val_score} to {val_score}. Saving model..."
            )
            best_val_score = val_score
            torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
            patience = 0
        else:
            print(f"Score did not improve from {best_val_score}.")
            patience += 1
            if patience >= patience_limit:
                print("Early stopping triggered.")
                break

    print(f"Stage 1 finished. Best Validation Score: {best_val_score}")
    return best_val_score
