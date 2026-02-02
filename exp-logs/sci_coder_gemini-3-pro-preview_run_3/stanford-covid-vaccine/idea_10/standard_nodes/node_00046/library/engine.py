import torch
import torch.nn as nn
import numpy as np
import os
import library.config
from library.utils import mcrmse_loss, seed_everything
from library.model import BiGRURegressor


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        reg_out = model(inputs)

        # Score only the valid positions for loss calculation
        reg_out_scored = reg_out[:, : library.config.Config.SEQ_SCORED, :]
        targets_scored = targets[:, : library.config.Config.SEQ_SCORED, :]

        loss = mcrmse_loss(reg_out_scored, targets_scored)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the Global MCRMSE score.
    Cite solution_lesson_node_00045: Correct Aggregation of Non-Linear Metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            reg_out = model(inputs)

            # Score only the valid positions
            reg_out_scored = reg_out[:, : library.config.Config.SEQ_SCORED, :]
            targets_scored = targets[:, : library.config.Config.SEQ_SCORED, :]

            # Filter for scored columns ONLY (reactivity, deg_Mg_pH10, deg_Mg_50C)
            reg_out_scored = reg_out_scored[
                :, :, library.config.Config.SCORED_COLS_INDICES
            ]
            targets_scored = targets_scored[
                :, :, library.config.Config.SCORED_COLS_INDICES
            ]

            all_preds.append(reg_out_scored)
            all_targets.append(targets_scored)

    # Concatenate all batches to compute global metric
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate global metric
    global_mcrmse = mcrmse_loss(all_preds, all_targets)

    return global_mcrmse.item()


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    seed_everything(library.config.Config.SEED)
    device = torch.device(library.config.Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = BiGRURegressor()
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=library.config.Config.LEARNING_RATE,
        weight_decay=library.config.Config.WEIGHT_DECAY,
    )

    # Initialize Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=library.config.Config.EPOCHS,
        eta_min=library.config.Config.SCHEDULER_MIN_LR,
    )

    # Training Loop Variables
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(library.config.Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, device)

        # Validate
        val_mcrmse = eval_fn(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{library.config.Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), library.config.Config.MODEL_SAVE_PATH)
            print(f"New best model saved with MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= library.config.Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")
    return best_mcrmse
