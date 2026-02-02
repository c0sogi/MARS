import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from library.config import (
    SEED,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    DEVICE,
    NUM_WORKERS,
    NUM_FOLDS,
    BACKBONE_NAME,
)
from library.utils import seed_everything
from library.dataset import SlabDataset, get_transforms
from library.model import WITSNetwork


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(probs)

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Handle edge case where batch might have only 1 class
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
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

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


def run_fold(fold, train_idx, val_idx, metadata_df):
    """
    Runs training for a single fold.
    """
    print(f"\n{'='*20} Fold {fold} {'='*20}")

    # 1. Prepare Datasets
    # We instantiate two datasets pointing to the same cache ("train_all")
    # One with train transforms, one with val transforms
    train_ds_full = SlabDataset(
        metadata_df,
        transform=get_transforms("train"),
        load_cached_data=True,
        split_name="train_all",
    )

    val_ds_full = SlabDataset(
        metadata_df,
        transform=get_transforms("val"),
        load_cached_data=True,
        split_name="train_all",
    )

    # Map Subject IDs to Dataset Indices
    # The dataset contains 3 slabs per subject. We must select all slabs corresponding to the split subjects.
    train_subjects = metadata_df.iloc[train_idx]["BraTS21ID"].values
    val_subjects = metadata_df.iloc[val_idx]["BraTS21ID"].values

    # dataset.ids contains the subject ID for each slab
    train_indices = np.where(np.isin(train_ds_full.ids, train_subjects))[0]
    val_indices = np.where(np.isin(val_ds_full.ids, val_subjects))[0]

    train_subset = Subset(train_ds_full, train_indices)
    val_subset = Subset(val_ds_full, val_indices)

    print(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Setup Model & Training
    model = WITSNetwork()
    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.16f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! AUC: {best_auc:.16f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_auc


def run_training():
    """
    Main entry point to run the cross-validation training loop.
    """
    seed_everything(SEED)

    # Load Metadata
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {TRAIN_METADATA_PATH}")

    metadata_df = pd.read_csv(TRAIN_METADATA_PATH)

    # GroupKFold
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    groups = metadata_df["BraTS21ID"]

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(metadata_df, metadata_df["MGMT_value"], groups)
    ):
        score = run_fold(fold, train_idx, val_idx, metadata_df)
        fold_scores.append(score)

    print(f"\n{'='*40}")
    print(f"Cross-Validation Complete")
    print(f"Folds: {fold_scores}")
    print(f"Mean AUC: {np.mean(fold_scores):.6f}")
    print(f"{'='*40}")
