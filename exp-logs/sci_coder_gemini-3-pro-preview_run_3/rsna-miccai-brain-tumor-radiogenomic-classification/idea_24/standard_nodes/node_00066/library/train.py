import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config, seed_everything
from library.utils import get_device
from library.data_loader import get_dataloaders
from library.model import RMSHDNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device).unsqueeze(1)  # Ensure target shape is (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(data)
        loss = criterion(logits, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * data.size(0)

        # Store predictions (apply sigmoid for probability) and targets for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(target.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC
    # Handle edge case where batch might contain only one class
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1)

            logits = model(data)
            loss = criterion(logits, target)

            running_loss += loss.item() * data.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(target.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training(epochs=Config.EPOCHS, learning_rate=Config.LEARNING_RATE):
    """
    Orchestrates the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data
    # get_dataloaders handles caching internally
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = RMSHDNet().to(device)

    # 4. Optimizer & Loss
    # Weight decay is disabled (0.0) as per configuration
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train & Validate
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        end_time = time.time()
        duration = end_time - start_time

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch + 1}/{epochs} | Time: {duration:.2f}s")
        print(f"  Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"  Val Loss:   {val_loss} | Val AUC:   {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  [Saved Best Model] New Best Val AUC: {best_val_auc}")
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")
    print(f"Best model saved to: {Config.MODEL_SAVE_PATH}")
