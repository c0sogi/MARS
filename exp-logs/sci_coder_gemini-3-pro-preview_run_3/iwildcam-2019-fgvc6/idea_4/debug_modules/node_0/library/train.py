import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_score, make_weighted_sampler
from library.dataset import AnimalDataset, get_transforms
from library.model import AnimalModel


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.scale(optimizer).step()
        scaler.update()

        # Scheduler Step (OneCycleLR updates every batch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro F1 score.
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Collect predictions for F1 score
            # outputs are logits, take argmax
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            batch_targets = labels.cpu().numpy()

            preds.extend(batch_preds)
            targets.extend(batch_targets)

    epoch_loss = running_loss / dataset_size
    epoch_f1 = get_score(targets, preds)

    return epoch_loss, epoch_f1


def run_training(debug=False, epochs=Config.EPOCHS, patience=4):
    """
    Main training function.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
        epochs (int): Number of training epochs.
        patience (int): Early stopping patience.

    Returns:
        float: Best Validation F1 score.
    """
    seed_everything(Config.SEED)

    # --- Data Loading ---
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        print("Debug mode active: limiting dataset size.")
        train_df = train_df.head(500)
        val_df = val_df.head(100)
        epochs = 2

    # Datasets
    train_dataset = AnimalDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = AnimalDataset(
        val_df, transforms=get_transforms("valid"), mode="valid"
    )

    # Weighted Sampler for Class Imbalance
    # Note: When using a sampler, shuffle must be False in DataLoader
    train_sampler = make_weighted_sampler(train_df, target_col="Category")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    device = torch.device(Config.DEVICE)
    model = AnimalModel(pretrained=True)
    model.to(device)

    # --- Optimization ---
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    # --- Training Loop ---
    print(f"Starting training: Epochs={epochs}, Batch Size={Config.BATCH_SIZE}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    best_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train & Validate
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )
        val_loss, val_f1 = valid_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F1: {val_f1:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_f1 > best_f1:
            print(
                f"Validation F1 improved ({best_f1:.10f} -> {val_f1:.10f}). Saving model..."
            )
            best_f1 = val_f1
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            print("Early stopping triggered. Training finished.")
            break

    print(f"Training complete. Best Validation F1: {best_f1:.10f}")
    return best_f1
