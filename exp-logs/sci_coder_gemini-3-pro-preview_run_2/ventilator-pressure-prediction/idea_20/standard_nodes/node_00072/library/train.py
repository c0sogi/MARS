import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, compute_mae
from library.loss import WeightedL1Loss
from library.model import CWCDP_BiLSTM
from library.dataset import load_data


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Calculate loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """
    Validation loop. Computes Loss and Inspiratory MAE.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass
            preds = model(x)

            # Calculate Loss (Weighted L1)
            loss = criterion(preds, y, u_out)

            # Calculate Metric (Inspiratory MAE)
            mae = compute_mae(preds, y, u_out)

            total_loss += loss.item()
            total_mae += mae
            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def train_model(debug=Config.DEBUG, load_cached_data=True):
    """
    Main function to train the CWCDP-BiLSTM model.

    Args:
        debug (bool): If True, runs on a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training on device: {device}")
    print(f"Debug Mode: {debug}")
    print(f"Stretched Horizon: {Config.EPOCHS} epochs")

    # 2. Data Loading
    train_dataset = load_data("train", debug=debug, load_cached_data=load_cached_data)
    val_dataset = load_data("val", debug=debug, load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization
    model = CWCDP_BiLSTM().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Stretched-Horizon Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = WeightedL1Loss(
        inspiratory_weight=Config.LOSS_INSPIRATORY_WEIGHT,
        expiratory_weight=Config.LOSS_EXPIRATORY_WEIGHT,
    )

    # 5. Training Loop
    best_mae = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Logging (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MAE: {val_mae}"
        )

        # Save Best Model
        if val_mae < best_mae:
            print(f"New Best MAE! ({best_mae} -> {val_mae}). Saving model...")
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Training complete. Best Validation MAE: {best_mae}")
    print(f"Best model saved to: {Config.MODEL_PATH}")
