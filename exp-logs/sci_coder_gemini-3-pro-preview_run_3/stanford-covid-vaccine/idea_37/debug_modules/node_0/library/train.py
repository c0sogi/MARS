import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.data import get_dataloaders
from library.model import DeepDecoupledModel


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, pair_indices, pair_mask, targets, ids) in enumerate(loader):
        inputs = inputs.to(device)
        pair_indices = pair_indices.to(device)
        pair_mask = pair_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_indices, pair_mask)

        # Calculate loss (MCRMSE on all 5 targets for training)
        loss = criterion(preds, targets, mode="train")

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device, config):
    """
    Evaluates the model on the validation set.
    Aggregates predictions first, then calculates MCRMSE on scored columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, pair_indices, pair_mask, targets, ids in loader:
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)
            pair_mask = pair_mask.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(inputs, pair_indices, pair_mask)

            # Collect results (move to CPU to save GPU memory during accumulation)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    if not all_preds:
        return 0.0

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    # Move back to device for metric calculation if necessary, or keep on CPU
    # (MCRMSE handles device matching, but usually metrics are fast enough on CPU)
    # Here we move to device to match the criterion's expectation if it has buffers,
    # though MCRMSE is stateless.
    cat_preds = cat_preds.to(device)
    cat_targets = cat_targets.to(device)

    # Calculate Metric (MCRMSE on scored columns only: reactivity, deg_Mg_pH10, deg_Mg_50C)
    # The criterion handles slicing to seq_scored (68) internally.
    val_score = criterion(cat_preds, cat_targets, mode="val")

    return val_score.item()


def train_model(config=None):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    if config is None:
        config = Config()

    # Ensure reproducibility
    set_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(config, load_cached_data=True)

    # Model
    model = DeepDecoupledModel().to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min
    )

    # Criterion
    criterion = MCRMSE()

    # Training State
    best_val_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )

        # Validate
        val_score = validate(model, val_loader, criterion, device, config)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.10f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Model Saving
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  >>> New Best Model Saved! Score: {best_val_score:.10f}")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Score: {best_val_score:.10f}")
    return best_val_score
