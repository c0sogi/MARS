import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import FractureDataset
from library.model import FractureModel
from library.utils import seed_everything


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # Input shape: (Batch, Slices, Channels, H, W)
        # Output shape: (Batch, Num_Classes)
        logits = model(images)

        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Store predictions and labels for potential metric calculation if needed
    # For now, we focus on the loss as the primary metric for early stopping

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, patience=5
):
    """
    Main training function handling data loading, model setup, and the training loop.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        debug (bool): If True, subsets data for quick debugging.
        patience (int): Early stopping patience.
    """
    seed_everything(Config.SEED)

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(
            f"DEBUG Mode: Training on {len(train_df)} samples, Validating on {len(val_df)} samples."
        )

    # 2. Datasets & Dataloaders
    train_dataset = FractureDataset(
        train_df,
        mode="train",
        load_cached_data=True,  # Enable caching to speed up repeated access
    )
    val_dataset = FractureDataset(val_df, mode="val", load_cached_data=True)

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

    # 3. Model Setup
    device = torch.device(Config.DEVICE)
    model = FractureModel(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    )
    model = model.to(device)

    # 4. Loss Function (BCE)
    # Cite solution_lesson_node_00001: Avoid Aggressive Class Weighting with Max-Pooling MIL
    criterion = nn.BCEWithLogitsLoss()

    # 5. Optimizer & Scheduler
    optimizer = Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. Training Loop with Early Stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{epochs} - Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
