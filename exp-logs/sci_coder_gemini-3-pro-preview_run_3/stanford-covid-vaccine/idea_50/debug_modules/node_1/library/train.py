import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, compute_mcrmse, AverageMeter
from library.data import get_dataloaders
from library.model import SDBR_BiGRU


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move data to device
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(features, pair_indices, pair_masks)

        # Calculate loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), features.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Performs validation and calculates MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].cpu().numpy()

            # Forward pass
            preds = model(features, pair_indices, pair_masks)

            # Collect predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    # scored_only=True for model selection
    val_score = compute_mcrmse(all_preds, all_targets, scored_only=True)

    # Also calculate full score for logging
    full_score = compute_mcrmse(all_preds, all_targets, scored_only=False)

    return val_score, full_score


def train_model(
    epochs=Config.EPOCHS,
    subset_size=Config.SUBSET_SIZE,
    batch_size=Config.BATCH_SIZE,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    max_grad_norm=Config.MAX_GRAD_NORM,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Main training function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Update Config for subsetting if provided
    # This is necessary because get_dataloaders reads Config.SUBSET_SIZE directly
    Config.SUBSET_SIZE = subset_size

    print(f"Starting training on device: {device}")
    print(f"Epochs: {epochs}, Subset Size: {subset_size}, Batch Size: {batch_size}")

    # 2. Data Loading
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=batch_size, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    model = SDBR_BiGRU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # 5. Loss Function
    criterion = MCRMSELoss()

    # 6. Training Loop
    best_val_score = float("inf")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )

        # Validate
        val_score, full_val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch + 1}/{epochs} | LR: {current_lr} | Train Loss: {train_loss} | Val MCRMSE (Scored): {val_score} | Val MCRMSE (All): {full_val_score}"
        )

        # Save Best Model
        if val_score < best_val_score:
            print(
                f"Validation score improved from {best_val_score} to {val_score}. Saving model to {save_path}"
            )
            best_val_score = val_score
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best Validation MCRMSE: {best_val_score}")
    return best_val_score
