import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    CACHE_DIR,
    NUM_WORKERS,
    SEED,
)
from library.utils import set_seed
from library.model import TMSVNet
from library.data import get_datasets


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move inputs to device
        flair = batch["flair"].to(device)
        t1wce = batch["t1wce"].to(device)
        t2w = batch["t2w"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(flair, t1wce, t2w)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * flair.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions by Subject ID to compute subject-level ROC AUC.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            flair = batch["flair"].to(device)
            t1wce = batch["t1wce"].to(device)
            t2w = batch["t2w"].to(device)
            targets = batch["target"].to(device)
            ids = batch["BraTS21ID"]

            logits = model(flair, t1wce, t2w)
            loss = criterion(logits, targets)

            running_loss += loss.item() * flair.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    val_loss = running_loss / len(dataloader.dataset)

    # Aggregate by Subject ID for metric calculation
    df_results = pd.DataFrame(
        {"BraTS21ID": all_ids, "target": all_targets, "pred": all_preds}
    )

    # Group by ID: mean prediction, max target (targets are constant per subject)
    df_grouped = df_results.groupby("BraTS21ID").agg({"target": "max", "pred": "mean"})

    try:
        val_auc = roc_auc_score(df_grouped["target"], df_grouped["pred"])
    except ValueError:
        # Handle edge cases (e.g., only one class in batch)
        val_auc = 0.5

    return val_loss, val_auc


def run_training():
    """
    Main execution function for training the TMSV-Net.
    """
    # 1. Setup
    set_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_save_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Device: {DEVICE}")
    print(f"Model Save Path: {model_save_path}")

    # 2. Data Loading
    # get_datasets handles the caching and expansion logic internally
    train_dataset, val_dataset, _ = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = TMSVNet().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved (AUC: {best_auc:.6f})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")
    return best_auc
