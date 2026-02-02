import os
import time
import torch
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed, compute_dice_score
from library.model import AttentionUNet25D
from library.loss import TverskyLoss
from library.dataset import get_loaders


def train_one_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        loss_fn: The loss function.
        device: The computation device.
        max_batches: Optional integer to limit the number of batches (for debugging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, loss_fn, device, max_batches=None):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        loss_fn: The loss function.
        device: The computation device.
        max_batches: Optional integer to limit the number of batches.

    Returns:
        tuple: (Average Loss, Average Dice Score)
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    count = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            # Calculate Dice Score
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            # Threshold to get binary predictions
            preds = (probs > Config.THRESHOLD).float()

            # Convert to numpy for the utility function
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # compute_dice_score calculates global dice for the flattened arrays
            batch_dice = compute_dice_score(preds_np, masks_np)

            running_loss += loss.item()
            running_dice += batch_dice
            count += 1

    avg_loss = running_loss / count if count > 0 else 0.0
    avg_dice = running_dice / count if count > 0 else 0.0

    return avg_loss, avg_dice


def run_training(debug=False, load_cached_data=True):
    """
    Orchestrates the training process.

    Args:
        debug (bool): If True, limits the dataset size for faster debugging.
        load_cached_data (bool): If True, attempts to load processed metadata from cache.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    train_loader, val_loader = get_loaders(load_cached_data=load_cached_data)

    # Determine batch limits for debugging
    max_train_batches = None
    max_val_batches = None
    if debug:
        # Calculate approximate batches based on DEBUG_SAMPLE_SIZE
        max_train_batches = max(1, Config.DEBUG_SAMPLE_SIZE // Config.BATCH_SIZE)
        max_val_batches = max(1, (Config.DEBUG_SAMPLE_SIZE // 4) // Config.BATCH_SIZE)
        print(
            f"Debug mode: Train batches={max_train_batches}, Val batches={max_val_batches}"
        )

    # 3. Model Initialization
    model = AttentionUNet25D()
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    loss_fn = TverskyLoss(
        alpha=Config.TVERSKY_ALPHA,
        beta=Config.TVERSKY_BETA,
        smooth=Config.TVERSKY_SMOOTH,
    )

    # 5. Training Loop
    best_dice = -1.0
    patience = 5
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_batches=max_train_batches,
        )

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, loss_fn, device, max_batches=max_val_batches
        )

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Time: {elapsed}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")

        # Early Stopping and Checkpointing
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model to {Config.MODEL_SAVE_PATH}"
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in Validation Dice. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Dice: {best_dice}")
