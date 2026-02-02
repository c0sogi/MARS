import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    BEST_MODEL_PATH,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
)
from library.utils import seed_everything
from library.data import get_dataloader
from library.model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect predictions for AUC
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle potential edge case where batch has only one class
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training(num_epochs=NUM_EPOCHS, debug=False):
    """
    Main training loop with Early Stopping.

    Args:
        num_epochs (int): Number of epochs to train.
        debug (bool): If True, uses a small subset of data for quick testing.
    """
    seed_everything(SEED)

    # 1. Load Metadata
    if not os.path.exists(TRAIN_METADATA_PATH) or not os.path.exists(VAL_METADATA_PATH):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)

    # Debug mode: subset data
    if debug:
        print("DEBUG MODE: Using subset of data.")
        df_train = df_train.head(32)
        df_val = df_val.head(32)

    # 2. Data Loaders
    # Note: caching is handled internally by BraTSDataset in get_dataloader
    train_loader = get_dataloader(df_train, phase="train")
    val_loader = get_dataloader(df_val, phase="val")

    # 3. Model Setup
    model = get_model()
    model = model.to(DEVICE)

    # 4. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # 5. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs on {DEVICE}...")

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f}")
        # Print full precision for validation AUC
        print(f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
