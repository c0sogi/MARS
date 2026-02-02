import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    OUTPUT_DIR,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from library.utils import seed_everything, AverageMeter, RocAucMeter
from library.dataset import get_loaders
from library.model import WideResNetECA


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run on (cuda/cpu).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, labels, _ in loader:
        images = images.to(device)
        # BCEWithLogitsLoss expects target shape (N, 1) to match output (N, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    losses = AverageMeter()
    auc_meter = RocAucMeter()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            # Apply sigmoid to convert logits to probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))
            auc_meter.update(labels, probs)

    return losses.avg, auc_meter.score()


def run_training(
    seed=42,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    load_cached_data=True,
    limit=None,
    patience=5,
):
    """
    Runs the full training pipeline for a specific seed.

    Args:
        seed (int): Random seed for reproducibility.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for the optimizer.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        limit (int, optional): Limit dataset size for debugging.
        patience (int): Early stopping patience.

    Returns:
        float: The best validation AUC achieved.
    """
    # 1. Set Seed
    seed_everything(seed)
    print(f"--- Starting Training | Seed: {seed} | Device: {DEVICE} ---")

    # 2. Get Data Loaders
    train_loader, val_loader, _ = get_loaders(
        batch_size=batch_size, load_cached_data=load_cached_data, limit=limit
    )

    # 3. Initialize Model
    model = WideResNetECA()
    model = model.to(DEVICE)

    # 4. Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training Loop
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Ensure output directory exists (handled in config, but good practice)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, f"model_seed_{seed}.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        # Update Scheduler
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  >>> New Best AUC! Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  >>> Early stopping triggered. No improvement for {patience} epochs."
                )
                break

    total_duration = time.time() - start_time
    print(
        f"Training Finished. Best AUC: {best_auc} (Epoch {best_epoch}). Total Time: {total_duration:.2f}s"
    )

    return best_auc
