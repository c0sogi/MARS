import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.data import get_data_loaders
from library.model import GranularSiameseDeBERTa


def train_fn(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    # Model outputs probabilities via sigmoid, so we use BCELoss
    loss_fn = nn.BCELoss()

    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        # Move batch data to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        q_segment_ids = batch["q_segment_ids"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        cats = batch["cats"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        preds = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            q_segment_ids=q_segment_ids,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
            cats=cats,
        )

        # Compute loss
        loss = loss_fn(preds, labels)

        # Scale loss for gradient accumulation
        loss = loss / Config.ACCUMULATION_STEPS
        loss.backward()

        if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0 or (
            batch_idx + 1
        ) == num_batches:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        total_loss += loss.item() * Config.ACCUMULATION_STEPS

    avg_loss = total_loss / num_batches
    return avg_loss


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on a given dataloader.
    Returns average loss, predictions, and targets (if available).
    """
    model.eval()
    total_loss = 0.0
    loss_fn = nn.BCELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            q_segment_ids = batch["q_segment_ids"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            cats = batch["cats"].to(device)

            preds = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                q_segment_ids=q_segment_ids,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
                cats=cats,
            )

            if "labels" in batch:
                labels = batch["labels"].to(device)
                loss = loss_fn(preds, labels)
                total_loss += loss.item()
                all_targets.append(labels.cpu().numpy())

            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    final_preds = np.concatenate(all_preds, axis=0)
    final_targets = np.concatenate(all_targets, axis=0) if all_targets else None

    return avg_loss, final_preds, final_targets


def main(epochs=Config.EPOCHS, debug=False):
    """
    Main training and evaluation pipeline.
    """
    # Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    if debug:
        epochs = 1
        print("Debug mode enabled: Epochs set to 1")

    # Model Initialization
    print("Initializing model...")
    model = GranularSiameseDeBERTa()
    model.to(device)

    # Optimizer with Differential Learning Rates
    # Group 1: Backbone (lower LR)
    # Group 2: Head / Custom Layers (higher LR)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    # Scheduler
    # Adjust steps for gradient accumulation
    num_update_steps_per_epoch = (
        len(train_loader) + Config.ACCUMULATION_STEPS - 1
    ) // Config.ACCUMULATION_STEPS
    num_training_steps = num_update_steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training Loop
    best_score = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 0

    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_preds, val_targets = eval_fn(model, val_loader, device)

        val_score = compute_spearmanr(val_preds, val_targets)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.0f}s - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Spearman: {val_score}"
        )

        # Checkpointing
        if val_score > best_score:
            print(
                f"Validation score improved ({best_score} -> {val_score}). Saving model..."
            )
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                best_model_wts, os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
            patience = 0
        else:
            patience += 1
            print(
                f"No improvement. Patience: {patience}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load Best Model for Inference
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    # Inference on Test Set
    print("Generating predictions on test set...")
    _, test_preds, _ = eval_fn(model, test_loader, device)

    # Create Submission
    print("Saving submission...")
    submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Ensure prediction shape matches submission
    if test_preds.shape[0] != len(submission):
        print(
            f"Warning: Prediction count {test_preds.shape[0]} differs from submission rows {len(submission)}."
        )

    submission[Config.TARGET_COLS] = test_preds
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
