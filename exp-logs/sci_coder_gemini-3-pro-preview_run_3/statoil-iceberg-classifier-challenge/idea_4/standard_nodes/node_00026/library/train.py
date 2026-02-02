import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    N_FOLDS,
    SEED,
    WORKING_DIR,
    DEVICE,
    TRAIN_META_CSV,
)
from library.utils import (
    AverageMeter,
    EarlyStopping,
    calculate_metric,
    set_seed,
)
from library.model import IcebergCNN
from library.data import (
    process_and_cache_data,
    IcebergDataset,
    get_transforms,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run training on (cpu or cuda).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on.

    Returns:
        tuple: (Average Loss, Log Loss Metric)
    """
    model.eval()
    losses = AverageMeter()

    # Store predictions and targets for metric calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Log Loss
    metric = calculate_metric(all_targets, all_preds)

    return losses.avg, metric


def run_fold(fold_idx):
    """
    Runs the training pipeline for a specific fold using Stratified K-Fold.

    Args:
        fold_idx (int): The index of the fold to train (0 to N_FOLDS-1).
    """
    # Ensure reproducibility
    set_seed(SEED)

    print(f"Loading data for Fold {fold_idx}...")
    # Load full training data
    (X_all, angle_all, y_all), _ = process_and_cache_data(load_cached_data=True)

    # Load metadata to define the training subset (80% of original data)
    # This ensures we do not train on the holdout validation set
    df_train_meta = pd.read_csv(TRAIN_META_CSV)
    train_indices_subset = df_train_meta["original_index"].values

    # Filter the full data to keep only the training subset
    X_train_subset = X_all[train_indices_subset]
    angle_train_subset = angle_all[train_indices_subset]
    y_train_subset = y_all[train_indices_subset]

    # Perform Stratified K-Fold Split on the subset
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Generate indices relative to the subset
    splits = list(skf.split(np.zeros(len(y_train_subset)), y_train_subset))
    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train_subset[train_idx],
        angle_train_subset[train_idx],
        y_train_subset[train_idx],
        transform=get_transforms("train"),
    )
    val_dataset = IcebergDataset(
        X_train_subset[val_idx],
        angle_train_subset[val_idx],
        y_train_subset[val_idx],
        transform=get_transforms("val"),
    )

    # Create DataLoaders
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

    # Initialize Model, Optimizer, Loss
    model = IcebergCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Model outputs sigmoid probabilities, so we use BCELoss
    criterion = nn.BCELoss()

    # Setup Checkpointing
    fold_dir = os.path.join(WORKING_DIR, f"fold_{fold_idx}")
    checkpoint_path = os.path.join(fold_dir, "model_best.pth")

    early_stopping = EarlyStopping(
        patience=PATIENCE, verbose=True, path=checkpoint_path
    )

    print(f"Starting training for Fold {fold_idx} on {DEVICE}...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metric = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val Metric (LogLoss): {val_metric:.6f}"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print(
        f"Fold {fold_idx} completed. Best Validation Loss: {early_stopping.val_loss_min:.6f}"
    )
