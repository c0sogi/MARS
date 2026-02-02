import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SiameseEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch_idx, (view_bulk, view_core, targets) in enumerate(loader):
        view_bulk = view_bulk.to(device)
        view_core = view_core.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Siamese forward pass
        logits = model(view_bulk, view_core)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Cite debug_lesson_1: Guard Metric Calculations Against Single-Class Data Subsets
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
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
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for view_bulk, view_core, targets in loader:
            view_bulk = view_bulk.to(device)
            view_core = view_core.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(view_bulk, view_core)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Cite debug_lesson_1: Guard Metric Calculations Against Single-Class Data Subsets
    if len(np.unique(all_targets)) < 2:
        val_auc = 0.5
    else:
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

    return val_loss, val_auc


def run_training(load_cached_data=True):
    """
    Main training pipeline.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed .npy files.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # get_dataloaders handles the caching logic internally based on the flag
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    model = SiameseEfficientNet()
    model.to(device)

    # 4. Optimization
    # Aggressive weight decay as per Idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    print(f"Model: {Config.MODEL_NAME} (Siamese Dual-Hypothesis)")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.10f} | Train AUC: {train_auc:.10f}")
        print(f"Val Loss:   {val_loss:.10f} | Val AUC:   {val_auc:.10f}")

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved ({best_val_auc:.10f} --> {val_auc:.10f}). Saving model..."
            )
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc:.10f}")
    print(f"Best model saved to: {Config.MODEL_PATH}")
