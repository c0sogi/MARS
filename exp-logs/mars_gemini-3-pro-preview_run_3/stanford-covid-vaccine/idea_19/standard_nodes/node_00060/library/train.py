import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.dataset import load_data, RNADataset
from library.model import DASR_BiGRU
from library.utils import MCRMSELoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Unpack batch
        # feat: (B, L, 14), pair_idx: (B, L), dist: (B, L), targets: (B, L, 5), mask: (B, L)
        feat, pair_idx, dist, targets, mask = batch

        feat = feat.to(device)
        pair_idx = pair_idx.to(device)
        dist = dist.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(feat, pair_idx, dist)

        # Slice predictions and targets to scored positions only (first 68)
        # This ensures the loss is calculated only on valid ground truth data.
        # The mask provided by dataset is 1.0 for [:seq_scored] and 0.0 otherwise,
        # but slicing is more direct for the provided MCRMSELoss implementation.
        # Cite debug_lesson_1: Align Metric Calculation with Scored Targets
        preds_scored = preds[:, : Config.SEQ_SCORED, Config.SCORED_COLS_INDICES]
        targets_scored = targets[:, : Config.SEQ_SCORED, Config.SCORED_COLS_INDICES]

        loss = criterion(preds_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Performs validation by aggregating predictions and calculating global MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            feat, pair_idx, dist, targets, mask = batch

            feat = feat.to(device)
            pair_idx = pair_idx.to(device)
            dist = dist.to(device)

            preds = model(feat, pair_idx, dist)

            # Move to CPU for aggregation
            all_preds.append(preds.cpu())
            all_targets.append(
                targets
            )  # targets are already CPU tensor from dataset usually, but let's be safe

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Slice to scored positions
    # Cite debug_lesson_1: Align Metric Calculation with Scored Targets
    preds_scored = all_preds[:, : Config.SEQ_SCORED, Config.SCORED_COLS_INDICES]
    targets_scored = all_targets[:, : Config.SEQ_SCORED, Config.SCORED_COLS_INDICES]

    # Calculate global metric
    loss = criterion(preds_scored, targets_scored)

    return loss.item()


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Main training function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = load_data("train", load_cached_data=load_cached_data, debug=debug)
    val_dataset = load_data("val", load_cached_data=load_cached_data, debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = DASR_BiGRU().to(device)

    # 4. Optimization
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=1e-6
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.GRAD_CLIP_NORM
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss} | "  # Full precision as requested
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
