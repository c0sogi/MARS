import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import (
    set_seed,
    AverageMeter,
    calculate_log_loss,
    predict_with_klein_tta,
)
from library.data import load_dataset, IcebergDataset, get_transforms
from library.model import IcebergResNet18


def apply_label_smoothing(labels, epsilon=0.05):
    """
    Applies label smoothing to binary labels.
    Formula: y_smooth = y * (1 - epsilon) + 0.5 * epsilon
    """
    return labels * (1 - epsilon) + 0.5 * epsilon


def train_one_epoch(model, loader, optimizer, device, epsilon):
    """
    Trains the model for one epoch using BCEWithLogitsLoss and Label Smoothing.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels, epsilon)

        optimizer.zero_grad()
        logits = model(images, angles)
        loss = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Validates the model using Klein Four-Group TTA and calculates Log Loss.
    """
    model.eval()
    preds_list = []
    targets_list = []

    # No gradient needed for validation
    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)

            # Predict with Test Time Augmentation
            # Returns probabilities of shape (B, 1)
            probs = predict_with_klein_tta(model, images, angles)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.numpy())

    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    # Calculate Log Loss
    # targets and preds are (N, 1)
    loss = calculate_log_loss(targets, preds)
    return loss


def run_calibration_phase():
    """
    Executes Phase 1: Calibration.
    Runs Stratified 5-Fold CV to determine the optimal convergence epoch.

    Returns:
        int: The average optimal convergence epoch (E_conv).
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Calibration Phase on {device}...")

    # 1. Load and Aggregate Data
    # We load both train and val splits from the metadata/cache and combine them
    # to perform a full Stratified K-Fold on the entire labeled dataset.
    print("Loading and aggregating datasets...")
    ds_train_part = load_dataset("train", load_cached_data=True)
    ds_val_part = load_dataset("val", load_cached_data=True)

    all_images = np.concatenate([ds_train_part.images, ds_val_part.images], axis=0)
    all_angles = np.concatenate([ds_train_part.angles, ds_val_part.angles], axis=0)
    all_labels = np.concatenate([ds_train_part.labels, ds_val_part.labels], axis=0)

    print(f"Total labeled samples: {len(all_labels)}")

    # 2. Stratified K-Fold Setup
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    best_epochs = []

    # 3. Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"\n=== Calibration Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Prepare Data Subsets
        train_imgs = all_images[train_idx]
        train_angs = all_angles[train_idx]
        train_lbls = all_labels[train_idx]

        val_imgs = all_images[val_idx]
        val_angs = all_angles[val_idx]
        val_lbls = all_labels[val_idx]

        # Create Datasets with appropriate transforms
        train_subset = IcebergDataset(
            train_imgs, train_angs, train_lbls, transform=get_transforms("train")
        )
        val_subset = IcebergDataset(
            val_imgs, val_angs, val_lbls, transform=get_transforms("val")
        )

        # DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model, Optimizer, Scheduler
        model = IcebergResNet18().to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR_BASE, weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.PHASE1_FACTOR,
            patience=Config.PHASE1_PATIENCE,
            verbose=True,
        )

        fold_best_loss = float("inf")
        fold_best_epoch = 0

        # Epoch Loop
        for epoch in range(1, Config.PHASE1_MAX_EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, Config.LABEL_SMOOTHING
            )
            val_loss = validate(model, val_loader, device)

            # Update scheduler
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:02d}: Train Loss: {train_loss:.6f}, TTA Val Loss: {val_loss:.10f}"
            )

            # Track best epoch
            if val_loss < fold_best_loss:
                fold_best_loss = val_loss
                fold_best_epoch = epoch

        print(
            f"Fold {fold + 1} Best Epoch: {fold_best_epoch} (Loss: {fold_best_loss:.6f})"
        )
        best_epochs.append(fold_best_epoch)

    # 4. Aggregate Results
    avg_best_epoch = int(np.round(np.mean(best_epochs)))
    print("\n=== Calibration Complete ===")
    print(f"Best Epochs per fold: {best_epochs}")
    print(f"Recommended Convergence Epoch (E_conv): {avg_best_epoch}")

    return avg_best_epoch
