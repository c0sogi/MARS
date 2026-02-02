import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, MCRMSELoss, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, device, max_batches=None
):
    """
    Executes one training epoch: Forward pass, Loss, Backprop, Gradient Clipping, Optimizer Step.
    Steps the scheduler at the end of the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        preds = model(inputs, pair_indices, pair_masks)
        loss = criterion(preds, targets)

        loss.backward()

        # Strict gradient clipping as per requirements
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    # Step the scheduler (CosineAnnealingLR T_max=EPOCHS implies per-epoch stepping)
    if scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device, max_batches=None):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally before calculating the metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, pair_indices, pair_masks)

            # Move to CPU for global aggregation
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

    if not all_preds:
        return 0.0

    # Global aggregation to avoid batch-averaging bias
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metric on scored columns only using the provided utility
    score = metric_mcrmse(all_preds, all_targets)

    return score


def run_training(epochs=Config.EPOCHS, patience=10, debug=False):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load DataLoaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE)

    # Debug configuration
    max_batches = 10 if debug else None
    if debug:
        epochs = 2
        print("Debug mode enabled: Training for 2 epochs with limited batches.")

    # Initialize Model
    model = RNAModel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function
    criterion = MCRMSELoss()

    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            max_batches=max_batches,
        )

        # Validation Step
        val_mcrmse = validate(model, val_loader, device, max_batches=max_batches)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed}s | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New Best Model Saved! MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_mcrmse
