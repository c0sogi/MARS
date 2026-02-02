import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data import get_dataloaders
from library.model import SiameseDeberta


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    # CrossEntropyLoss supports soft targets (probabilities) in recent PyTorch versions
    criterion = nn.CrossEntropyLoss()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            logits = model(
                input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, features
            )
            loss = criterion(logits, targets)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and log loss metric.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            features = batch["features"].to(device)
            targets = batch["target"].to(device)

            with autocast():
                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    features,
                )
                loss = criterion(logits, targets)

            total_loss += loss.item()

            # Apply softmax to get probabilities for metric calculation
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Log Loss Metric
    metric_score = compute_log_loss(all_targets, all_preds)

    return avg_loss, metric_score


def run_training(debug: bool = False, load_cached_data: bool = True):
    """
    Main training loop with differential learning rates, early stopping, and model saving.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        debug=debug,
        load_cached_data=load_cached_data,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    print("Initializing model...")
    model = SiameseDeberta()
    model.to(device)

    # 4. Optimizer with Differential Learning Rates
    # Separate backbone parameters from head/pooling parameters
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
        eps=Config.EPS,
    )

    # 5. Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.NUM_WARMUP_STEPS_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 6. Mixed Precision Scaler
    scaler = GradScaler()

    # 7. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1}/{Config.EPOCHS} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val Metric: {val_metric}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            print(
                f"  Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(
                f"  Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation Loss: {best_val_loss}")
