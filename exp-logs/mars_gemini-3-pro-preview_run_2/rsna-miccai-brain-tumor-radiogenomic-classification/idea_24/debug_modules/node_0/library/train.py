import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    MODEL_NAME,
    PRETRAINED,
)
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass (logits)
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(probs)

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case with single class in batch/epoch (rare but possible)
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
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs)

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(load_cached_data=True, max_epochs=EPOCHS):
    """
    Main execution function for training the model.

    Args:
        load_cached_data (bool): Whether to use cached ROI indices.
        max_epochs (int): Override for number of epochs (useful for debugging).
    """
    # 1. Setup
    seed_everything()
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation is complete."
        )

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 3. Get DataLoaders
    # We pass None for test_df as we are only training here
    train_loader, val_loader, _ = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        load_cached_data=load_cached_data,
    )

    # 4. Initialize Model
    model = AsymmetricEfficientNet(model_name=MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device)

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {max_epochs} epochs...")

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{max_epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"  Val Loss:   {val_loss} | Val AUC:   {val_auc}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            print(
                f"  [New Best] AUC improved from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc
