import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_and_cache_data, IcebergDataset
from library.model import QPWBN

# Initialize logger
logger = get_logger("train")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, inc_angles, targets) in enumerate(loader):
        inputs = inputs.to(device)
        inc_angles = inc_angles.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, inc_angles)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for inputs, inc_angles, targets in loader:
            inputs = inputs.to(device)
            inc_angles = inc_angles.to(device)
            targets = targets.to(device)

            outputs = model(inputs, inc_angles)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_fold(fold_idx, train_idx, val_idx, X_full, y_full, inc_full, device):
    """
    Executes the training process for a single fold.
    """
    logger.info(f"Starting Fold {fold_idx + 1}/{Config.NUM_FOLDS}")

    # 1. Prepare Data for this Fold
    X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
    y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
    inc_train_fold, inc_val_fold = inc_full[train_idx], inc_full[val_idx]

    # Create Datasets
    # Apply augmentation only to training set
    train_dataset = IcebergDataset(
        X_train_fold, inc_train_fold, y_train_fold, transform=True
    )
    val_dataset = IcebergDataset(X_val_fold, inc_val_fold, y_val_fold, transform=False)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = QPWBN().to(device)

    # 3. Setup Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 4. Training Loop with Early Stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        # Debug mode shortcut
        if Config.DEBUG and epoch >= Config.DEBUG_EPOCHS:
            logger.info("Debug mode: Reached max debug epochs.")
            break

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging
        logger.info(
            f"Fold {fold_idx+1} Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model immediately to disk to be safe
            save_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold_idx}.pth")
            torch.save(best_model_wts, save_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info(f"Fold {fold_idx+1} Best Val Loss: {best_loss:.10f}")
    return best_loss


def run_training():
    """
    Main orchestration function for Stratified K-Fold Cross Validation.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load and Combine Data
    # We combine the metadata-defined 'train' and 'val' splits to form a full development set
    # for our own K-Fold splitting.
    data = process_and_cache_data(load_cached_data=True)

    X_train_part = data["X_train"]
    y_train_part = data["y_train"]
    inc_train_part = data["inc_train"]

    X_val_part = data["X_val"]
    y_val_part = data["y_val"]
    inc_val_part = data["inc_val"]

    # Concatenate
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    inc_full = np.concatenate([inc_train_part, inc_val_part], axis=0)

    logger.info(f"Full development set shape: {X_full.shape}")

    # Handle Debug Mode
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode active. Truncating data to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        limit = min(len(X_full), Config.DEBUG_SAMPLE_SIZE)
        X_full = X_full[:limit]
        y_full = y_full[:limit]
        inc_full = inc_full[:limit]

    # 2. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        best_fold_loss = run_fold(
            fold_idx, train_idx, val_idx, X_full, y_full, inc_full, device
        )
        cv_scores.append(best_fold_loss)

    # 3. Summary
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    logger.info("=" * 30)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 30)
    for i, score in enumerate(cv_scores):
        logger.info(f"Fold {i+1}: {score:.10f}")
    logger.info(f"Mean Log Loss: {mean_score:.10f} (+/- {std_score:.10f})")
    logger.info("=" * 30)
