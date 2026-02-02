import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU


class MCRMSELoss(nn.Module):
    """
    MCRMSE Loss for training.
    Computes Mean Columnwise Root Mean Squared Error over all 5 targets.
    Slices inputs to Config.SEQ_SCORED before calculation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Slice to scored sequence length (68)
        # Preds: (B, 107, 5) -> (B, 68, 5)
        # Targets: (B, 107, 5) -> (B, 68, 5)
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]
        targets_sliced = targets[:, : Config.SEQ_SCORED, :]

        # MSE per column: Mean over Batch and Sequence dimensions
        mse = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(0, 1))

        # RMSE per column
        rmse = torch.sqrt(mse)

        # Mean of RMSEs
        loss = torch.mean(rmse)
        return loss


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices, pair_mask)

        # Compute loss (MCRMSE on all 5 targets, sliced)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Validates the model on the validation set.
    Returns the official MCRMSE score using library.utils.compute_mcrmse.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU for aggregation

            outputs = model(features, pair_indices, pair_mask)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metric using the official utility
    # This handles slicing to 68 and filtering to the 3 scored columns
    score = compute_mcrmse(all_preds, all_targets)

    return score


def run_training():
    """
    Main driver function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.create_directories()
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(debug=Config.DEBUG)

    # 3. Model
    print("Initializing model...")
    model = DeepStabilizedBiGRU().to(device)

    # 4. Optimization
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss} | "
            f"Val MCRMSE: {val_score}"
        )

        # Early Stopping & Model Saving
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
