import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.loss import MCRMSELoss
from library.data import get_loader
from library.model import DIN_CG_BiGRU


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Move batch to device
        features = batch["features"].to(device)
        bpps_indices = batch["bpps_indices"].to(device)
        bpps_mask = batch["bpps_mask"].to(device)
        targets = batch["targets"].to(device)
        batch_size = features.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, bpps_indices, bpps_mask)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the global MCRMSE metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            bpps_indices = batch["bpps_indices"].to(device)
            bpps_mask = batch["bpps_mask"].to(device)
            targets = batch["targets"]  # Keep targets on CPU for aggregation

            # Forward pass
            outputs = model(features, bpps_indices, bpps_mask)

            # Move outputs to CPU and collect
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Global MCRMSE
    # The MCRMSE utility function handles slicing to Config.PRED_LEN internally
    score = MCRMSE(all_targets, all_preds)

    return score


def run_training():
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Device Setup
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 3. Data Loaders
    print("Initializing DataLoaders...")
    train_loader = get_loader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
    )
    val_loader = get_loader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = DIN_CG_BiGRU().to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 6. Loss Function
    criterion = MCRMSELoss()

    # 7. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting Training...")
    print("-" * 50)

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val MCRMSE: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"*** New Best Model Saved! Score: {best_score} ***")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered. Training finished.")
            break

    print("-" * 50)
    print(f"Training Complete. Best Validation MCRMSE: {best_score}")
