import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader
from library.model import WITSNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Aggregates slab-level predictions to subject-level predictions before computing AUC.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, ids in loader:
            images = images.to(device)
            targets_dev = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets_dev)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    avg_loss = running_loss / len(loader.dataset)

    # Aggregate predictions by Subject ID
    # Create a DataFrame for easy grouping
    df_results = pd.DataFrame(
        {"BraTS21ID": all_ids, "prob": all_preds, "target": all_targets}
    )

    # Group by ID and take the mean of probabilities
    # Targets are constant per subject, so mean/max/min works
    df_agg = (
        df_results.groupby("BraTS21ID")
        .agg({"prob": "mean", "target": "mean"})
        .reset_index()
    )

    # Calculate AUC
    try:
        auc = roc_auc_score(df_agg["target"], df_agg["prob"])
    except ValueError:
        # Handle case where only one class is present in batch/subset
        auc = 0.5

    return avg_loss, auc


def run_training():
    """
    Main training loop.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        print(
            f"DEBUG Mode: limiting training data to {Config.DEBUG_SUBSET_SIZE} subjects."
        )
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)

    # 3. Create DataLoaders
    train_loader = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = get_dataloader(
        df_val, mode="val", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print(f"Train instances (slabs): {len(train_loader.dataset)}")
    print(f"Val instances (slabs): {len(val_loader.dataset)}")

    # 4. Initialize Model
    print("Initializing WITS-Net...")
    model = WITSNet()
    model.to(device)

    # 5. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop with Early Stopping
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")
