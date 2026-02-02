import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from library.config import Config
from library.utils import set_seed, mcrmse_metric, get_scored_indices
from library.data import get_dataloaders
from library.model import RNARegressor


def loss_fn(outputs, targets):
    """
    Differentiable MCRMSE loss function for training.

    Args:
        outputs: (Batch, SeqLen, 5)
        targets: (Batch, SeqLen, 5)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Slice to scored sequence length (0 to 68)
    # The targets are only valid for the first 68 bases.
    outputs_sliced = outputs[:, : Config.SEQ_SCORED, :]
    targets_sliced = targets[:, : Config.SEQ_SCORED, :]

    # Calculate Squared Error: (y_hat - y)^2
    squared_error = (outputs_sliced - targets_sliced) ** 2

    # MSE per column: Average over Batch (0) and Sequence (1) dimensions
    mse_per_column = torch.mean(squared_error, dim=(0, 1))

    # RMSE per column: Sqrt(MSE)
    # Add a small epsilon to avoid NaN gradients if MSE is exactly 0
    rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

    # Average RMSE across all 5 columns (Multi-Task Learning)
    loss = torch.mean(rmse_per_column)

    return loss


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        adj_indices = batch["adjacency_indices"].to(device)
        adj_mask = batch["adjacency_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, adj_indices, adj_mask)

        # Compute loss
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    Computes MCRMSE on the specific scored columns globally.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            adj_indices = batch["adjacency_indices"].to(device)
            adj_mask = batch["adjacency_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU for accumulation

            outputs = model(inputs, adj_indices, adj_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Get indices for the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    scored_indices = get_scored_indices()

    # Compute Metric
    # Note: mcrmse_metric handles slicing to seq_scored internally
    score = mcrmse_metric(
        y_true, y_pred, seq_scored=Config.SEQ_SCORED, target_indices=scored_indices
    )

    return score


def run_training():
    """
    Main execution function for training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    model = RNARegressor().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val MCRMSE: {val_score} | "
            f"LR: {current_lr} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
