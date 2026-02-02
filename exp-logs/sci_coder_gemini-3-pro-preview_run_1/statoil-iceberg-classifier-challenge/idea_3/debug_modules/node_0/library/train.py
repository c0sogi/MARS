import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

from library.config import (
    N_FOLDS,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    PATIENCE,
    NUM_WORKERS,
    SEED,
    get_model_path,
)
from library.utils import seed_everything, get_device, save_checkpoint
from library.dataset import load_data, IcebergDataset, get_transforms
from library.model import IcebergEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.0):
    """
    Trains the model for one epoch.
    Applies label smoothing to the targets manually before passing to BCEWithLogitsLoss.
    """
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Apply label smoothing: y_new = y * (1 - eps) + 0.5 * eps
        # This smooths binary targets towards 0.5
        if label_smoothing > 0:
            with torch.no_grad():
                labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        # Reshape labels to (Batch, 1) to match model output
        labels = labels.view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Uses standard targets (no smoothing) for metric calculation.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def fit_fold(fold_idx, train_loader, val_loader, device):
    """
    Trains a single fold of the cross-validation.
    Handles early stopping and checkpoint saving.
    """
    print(f"\nStarting training for Fold {fold_idx + 1}/{N_FOLDS}")

    # Initialize model
    model = IcebergEfficientNet()
    model.to(device)

    # Optimizer with high weight decay as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Loss function (BCEWithLogitsLoss combines Sigmoid + BCE)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_smoothing=LABEL_SMOOTHING,
        )

        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision for validation loss
        print(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.16f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            # Save best model
            model_path = get_model_path(fold_idx)
            save_checkpoint(model, optimizer, epoch, val_loss, model_path)
            print(
                f"New best model saved for Fold {fold_idx + 1} with loss {best_loss:.6f}"
            )
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_loss


def run_training():
    """
    Main driver function to execute the Stratified K-Fold training pipeline.
    """
    # Set seed for reproducibility
    seed_everything(SEED)

    device = get_device()
    print(f"Using device: {device}")

    # Load Data (handles caching internally)
    print("Loading data...")
    images, angles, labels = load_data(mode="train", load_cached_data=True)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []

    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        # Subset data
        X_train, X_val = images[train_idx], images[val_idx]
        a_train, a_val = angles[train_idx], angles[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Create Datasets
        # Apply aggressive augmentation to training set
        train_dataset = IcebergDataset(
            X_train, a_train, y_train, transform=get_transforms(mode="train")
        )
        # Standard transform for validation
        val_dataset = IcebergDataset(
            X_val, a_val, y_val, transform=get_transforms(mode="valid")
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

        # Train the fold
        best_loss = fit_fold(fold_idx, train_loader, val_loader, device)
        fold_scores.append(best_loss)

    # Summary
    print("\nTraining Complete.")
    print("Fold Scores (Log Loss):")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.16f}")
    print(f"Average: {np.mean(fold_scores):.16f}")
