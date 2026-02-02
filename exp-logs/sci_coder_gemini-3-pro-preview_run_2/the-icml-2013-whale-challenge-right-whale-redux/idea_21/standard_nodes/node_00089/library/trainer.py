import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import (
    MetricMonitor,
    get_score,
    save_checkpoint,
    seed_everything,
)
from library.dataset import WhaleDataset, WhaleTransforms
from library.models import WhaleClassifier

# Ensure reproducibility
seed_everything(Config.SEED)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.avg_metrics["Loss"]


def valid_one_epoch(model, loader, criterion, device):
    """
    Performs validation on the validation set.
    Returns average loss, AUC score, and predictions.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            preds_list.extend(probs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    preds_array = np.array(preds_list).flatten()
    targets_array = np.array(targets_list).flatten()

    auc = get_score(targets_array, preds_array)

    return metric_monitor.avg_metrics["Loss"], auc, preds_array, targets_array


def run_fold(fold_idx, model_name, df):
    """
    Orchestrates the training for a specific fold.
    Implements Multi-Objective Checkpointing (Best AUC and Best Loss).
    """
    print(f"\n[Fold {fold_idx}] Starting training for model: {model_name}")

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG mode: Truncating dataframe to {Config.DEBUG_SAMPLES} samples.")
        df = df.iloc[: Config.DEBUG_SAMPLES].copy()

    # 1. Split Data (Stratified K-Fold)
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    train_idx, val_idx = list(skf.split(df, df["label"]))[fold_idx]
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # 2. Datasets
    # Pass unique split names to ensure caching works correctly per fold
    train_dataset = WhaleDataset(
        train_df,
        split_name=f"train_fold_{fold_idx}",
        transform=WhaleTransforms(mode="train"),
    )
    val_dataset = WhaleDataset(
        val_df, split_name=f"val_fold_{fold_idx}", transform=None
    )

    # 3. Weighted Random Sampler
    # Calculate weights to balance the batches
    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels)
    # Avoid division by zero if a class is missing (unlikely in stratified split)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model, Optimizer, Scheduler, Loss
    device = torch.device(Config.DEVICE)
    model = WhaleClassifier(model_name, pretrained=Config.PRETRAINED)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop with Multi-Objective Checkpointing
    best_auc = 0.0
    best_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_loss, val_auc, _, _ = valid_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Checkpoint: Best AUC
        if val_auc > best_auc:
            best_auc = val_auc
            filename = f"{model_name}_fold_{fold_idx}_best_auc.pth"
            save_checkpoint(model, optimizer, epoch, val_auc, filename)
            print(f"  --> Saved Best AUC Checkpoint: {val_auc:.6f}")

        # Checkpoint: Best Loss
        if val_loss < best_loss:
            best_loss = val_loss
            filename = f"{model_name}_fold_{fold_idx}_best_loss.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, filename)
            print(f"  --> Saved Best Loss Checkpoint: {val_loss:.6f}")

    print(
        f"[Fold {fold_idx}] Finished. Best AUC: {best_auc:.6f}, Best Loss: {best_loss:.6f}"
    )

    # Clean up to free GPU memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
