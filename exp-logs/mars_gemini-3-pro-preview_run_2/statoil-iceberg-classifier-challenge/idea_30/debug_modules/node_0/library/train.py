import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import EarlyStopping
from library.data_loader import IcebergDataset
from library.model import SC_WBN, train_one_epoch, validate


def run_fold(fold_idx, X_train, inc_train, y_train, X_val, inc_val, y_val, device):
    """
    Executes the training and validation pipeline for a single fold.

    Args:
        fold_idx (int): The index of the current fold (0-based).
        X_train (np.ndarray): Training images.
        inc_train (np.ndarray): Training incidence angles.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation images.
        inc_val (np.ndarray): Validation incidence angles.
        y_val (np.ndarray): Validation labels.
        device (torch.device): The device (CPU/GPU) to use for training.

    Returns:
        float: The best validation loss (BCE) achieved during the fold.
    """
    print(f"\nStarting Fold {fold_idx + 1}...")

    # 1. Prepare DataLoaders
    # Training dataset with augmentation
    train_dataset = IcebergDataset(X_train, inc_train, y_train, transform=True)
    # Validation dataset without augmentation
    val_dataset = IcebergDataset(X_val, inc_val, y_val, transform=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. Initialize Model and Optimization Components
    model = SC_WBN().to(device)
    criterion = nn.BCEWithLogitsLoss()

    # "Low and Slow" optimization
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="min")

    # 3. Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        # Optimization Step
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation Step
        val_loss, val_log_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val LogLoss: {val_log_loss} | Val Acc: {val_acc}"
        )

        # Early Stopping Check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 4. Save Best Model
    if early_stopping.best_model_state is not None:
        save_path = os.path.join(Config.WORKING_DIR, f"sc_wbn_fold_{fold_idx}.pth")
        torch.save(early_stopping.best_model_state, save_path)
        print(f"Saved best model for fold {fold_idx + 1} to {save_path}")

    return early_stopping.val_score_best
