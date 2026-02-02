import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_device, mcrmse_loss
from library.data import get_dataloaders
from library.model import DR_RHN


def train_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (x, y, mask, p_idx, p_mask, _) in enumerate(loader):
        # Move data to device
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        p_idx = p_idx.to(device)
        p_mask = p_mask.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns (y1, y2) corresponding to the two passes in the recycling loop
        y1, y2 = model(x, p_idx, p_mask)

        # Calculate Loss
        # We use the scored indices defined in Config for the metric
        # L_total = MCRMSE(y2) + 0.5 * MCRMSE(y1)
        loss_main = mcrmse_loss(y2, y, mask, Config.SCORED_TARGET_INDICES)
        loss_aux = mcrmse_loss(y1, y, mask, Config.SCORED_TARGET_INDICES)

        loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Runs validation loop.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y, mask, p_idx, p_mask, _ in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            p_idx = p_idx.to(device)
            p_mask = p_mask.to(device)

            # Forward pass
            # We only care about the final refined prediction y2 for validation
            _, y2 = model(x, p_idx, p_mask)

            # Calculate Metric
            loss = mcrmse_loss(y2, y, mask, Config.SCORED_TARGET_INDICES)
            running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def run_training(num_epochs=Config.EPOCHS, load_cached_data=True):
    """
    Main function to run the training pipeline.

    Args:
        num_epochs (int): Number of epochs to train.
        load_cached_data (bool): Whether to load preprocessed data from cache.
    """
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 2. Initialize Model
    print("Initializing DR-RHN Model...")
    model = DR_RHN().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_loss)

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val MCRMSE: {val_loss:.10f} | "
            f"Time: {duration:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best Model Saved! (Loss: {best_val_loss:.10f})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_val_loss:.10f}")
