import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger
from library.dataset import get_dataset
from library.model import IcebergResNet

logger = get_logger("Train")


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, epoch, label_smoothing=0.0
):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to train on.
        epoch: Current epoch number.
        label_smoothing: Float value for label smoothing (default 0.0).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, angles, labels, _) in enumerate(dataloader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (B, 1)

        batch_size = images.size(0)
        dataset_size += batch_size

        # Apply Label Smoothing
        # Target transformation: y_ls = y * (1 - epsilon) + 0.5 * epsilon
        if label_smoothing > 0.0:
            with torch.no_grad():
                labels_smoothed = (
                    labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
                )
        else:
            labels_smoothed = labels

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels_smoothed)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to validate on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch_idx, (images, angles, labels, _) in enumerate(dataloader):
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            batch_size = images.size(0)
            dataset_size += batch_size

            outputs = model(images, angles)

            # For validation, we use the raw labels (0 or 1) to calculate the metric
            # Note: criterion is BCEWithLogitsLoss, so it expects logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_fold(fold_idx=0):
    """
    Orchestrates the training for a single fold.

    Args:
        fold_idx (int): Index of the current fold (used for saving checkpoints).

    Returns:
        model: The trained model with the best validation weights.
    """
    logger.info(f"Starting training for Fold {fold_idx}...")

    device = torch.device(Config.DEVICE)

    # 1. Load Datasets
    # Using the library function to get datasets.
    # Note: In this setup, 'train' and 'val' splits are fixed by the metadata files.
    train_dataset = get_dataset("train", load_cached_data=True)
    val_dataset = get_dataset("val", load_cached_data=True)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = IcebergResNet()
    model = model.to(device)

    # 4. Optimizer and Scheduler
    # AdamW with weight decay as specified in Config
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Loss Function
    # BCEWithLogitsLoss is used. Label smoothing is applied manually in train_one_epoch.
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.10f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0

            # Save best model to disk
            save_path = os.path.join(
                Config.WORKING_DIR, f"model_fold_{fold_idx}_best.pth"
            )
            torch.save(best_model_wts, save_path)
            logger.info(f"New best model saved to {save_path}")
        else:
            early_stopping_counter += 1

        # Early Stopping
        if early_stopping_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"Training complete. Best Val Loss: {best_loss:.10f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    return model
