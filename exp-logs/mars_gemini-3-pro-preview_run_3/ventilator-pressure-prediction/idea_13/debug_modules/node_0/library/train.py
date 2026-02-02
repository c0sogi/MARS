import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, get_device, AverageMeter
from library.data import prepare_datasets
from library.model import TAPINNet


def masked_mae_loss(preds, targets, u_out):
    """
    Calculates L1 Loss (MAE) only for the inspiratory phase.
    The inspiratory phase is defined where u_out is approximately 0.
    """
    # Create boolean mask (u_out == 0)
    # Using < 0.5 is robust for float representations of binary flags
    mask = u_out < 0.5

    # Handle edge case where batch has no inspiratory phase (unlikely)
    if not mask.any():
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    # Calculate MAE only on masked elements
    return nn.functional.l1_loss(preds[mask], targets[mask])


def train_epoch(model, loader, optimizer, device, max_batches=None):
    """
    Runs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        # Extract u_out for masking (Index 1 in Config.MODEL_FEATURES)
        u_out = x[:, :, 1]

        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Compute loss
        loss = masked_mae_loss(preds, y, u_out)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), x.size(0))

    return loss_meter.avg


def validate_epoch(model, loader, device, max_batches=None):
    """
    Runs evaluation on the validation set.
    """
    model.eval()
    mae_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)
            u_out = x[:, :, 1]

            preds = model(x)

            mae = masked_mae_loss(preds, y, u_out)
            mae_meter.update(mae.item(), x.size(0))

    return mae_meter.avg


def train_model(
    epochs=Config.EPOCHS, load_cached_data=True, save_model=True, debug=False
):
    """
    Main function to train the TAPIN-Net model.

    Args:
        epochs (int): Number of training epochs.
        load_cached_data (bool): Whether to load pre-processed .npy files.
        save_model (bool): Whether to save the best model checkpoint.
        debug (bool): If True, runs on a small subset of data for testing.

    Returns:
        model: The trained PyTorch model (loaded with best weights).
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Initializing training on device: {device}")

    # 2. Data Loading
    # prepare_datasets handles caching internally
    train_loader, val_loader, _ = prepare_datasets(load_cached_data=load_cached_data)

    # Determine batch limits for debugging
    max_batches = 10 if debug else None

    # 3. Model Initialization
    model = TAPINNet().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # 5. Training Loop
    best_mae = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, max_batches=max_batches
        )
        val_mae = validate_epoch(model, val_loader, device, max_batches=max_batches)

        # Scheduler Step
        scheduler.step(val_mae)

        # Print metrics in full precision
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val MAE: {val_mae}"
        )

        # Checkpointing
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            if save_model:
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with MAE: {best_mae}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training complete. Best Val MAE: {best_mae}")

    # Load best weights before returning
    if save_model and os.path.exists(best_model_path):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
