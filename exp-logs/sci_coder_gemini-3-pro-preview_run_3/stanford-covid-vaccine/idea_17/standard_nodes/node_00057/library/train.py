import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import DISR_BiGRU


def train_one_epoch(model, loader, optimizer, config):
    """
    Performs one epoch of training using gradient clipping and masked MCRMSE loss.
    """
    model.train()
    train_loss_accum = 0.0

    for batch in loader:
        features = batch["features"].to(config.device)
        pair_indices = batch["pair_indices"].to(config.device)
        targets = batch["targets"].to(config.device)
        mask = batch["mask"].to(config.device)  # Shape: (B, L)

        optimizer.zero_grad()

        # Forward pass
        # preds shape: (B, L, 5)
        preds = model(features, pair_indices)

        # Apply mask to loss calculation
        # We only care about positions where mask == 1 (first 68 bases)
        # Expand mask for targets: (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1)

        # Mask predictions and targets
        preds_masked = preds * mask_expanded
        targets_masked = targets * mask_expanded

        # Calculate Loss (MCRMSE on all 5 columns)
        loss = mcrmse_loss(preds_masked, targets_masked)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        # Critical for stabilizing the deep recurrent architecture
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        # Optimizer step
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / len(loader)
    return avg_train_loss


def validate_global(model, loader, config):
    """
    Performs validation using Global Metric Aggregation.
    Concatenates all predictions and targets before calculating MCRMSE to avoid batch-averaging bias.
    Evaluates only on the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(config.device)
            pair_indices = batch["pair_indices"].to(config.device)
            targets = batch["targets"].to(config.device)
            mask = batch["mask"].to(config.device)

            preds = model(features, pair_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())

    # Concatenate all batches
    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds, dim=0)  # (N, L, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, L, 5)
    all_masks = torch.cat(all_masks, dim=0)  # (N, L)

    # Filter for Scored Columns only (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Indices correspond to config.target_cols:
    # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Indices: 0, 1, 3
    scored_indices = [0, 1, 3]

    all_preds_scored = all_preds[:, :, scored_indices]
    all_targets_scored = all_targets[:, :, scored_indices]

    # Apply Mask
    mask_expanded = all_masks.unsqueeze(-1)  # (N, L, 1)
    all_preds_scored = all_preds_scored * mask_expanded
    all_targets_scored = all_targets_scored * mask_expanded

    # Calculate Validation Metric
    val_loss = mcrmse_loss(all_preds_scored, all_targets_scored).item()

    return val_loss


def run_training(config: Config):
    """
    Main training loop implementing the full training pipeline with Early Stopping.
    """
    set_seed(config.seed)

    # Load DataLoaders
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize Model
    model = DISR_BiGRU(config).to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    # Training State
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {config.device}...")
    print(
        f"Model: DISR-BiGRU | Layers: {config.num_layers} | Hidden: {config.hidden_dim}"
    )

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, config)

        # Validate
        val_loss = validate_global(model, val_loader, config)

        # Update Scheduler
        scheduler.step()

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCRMSE (Scored): {val_loss:.10f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  [+] Saved best model to {config.model_save_path}")
        else:
            patience_counter += 1
            print(f"  [-] Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_val_loss:.10f}")
