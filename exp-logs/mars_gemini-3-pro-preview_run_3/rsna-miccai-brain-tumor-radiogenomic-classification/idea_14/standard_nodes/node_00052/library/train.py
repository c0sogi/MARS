import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_train_loader, get_val_loader
from library.model import BraTS25DNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Calculate AUC
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        # Handle edge case where only one class is present in the batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs)

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training():
    """
    Main training loop with Early Stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loaders
    # Using load_cached=True to utilize the caching mechanism in data_loader.py
    train_loader = get_train_loader(load_cached=True)
    val_loader = get_val_loader(load_cached=True)

    # 3. Model
    model = BraTS25DNet()
    model.to(device)

    # 4. Optimizer & Loss
    # Explicitly using Config parameters
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        end_time = time.time()
        duration = (end_time - start_time) / 60.0

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} | Time: {duration} min")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved from {best_val_auc} to {val_auc}. Saving model to {Config.MODEL_SAVE_PATH}"
            )
            best_val_auc = val_auc
            patience_counter = 0

            # Ensure directory exists
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_val_auc}")
    return best_val_auc
