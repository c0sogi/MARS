import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import AppleLeafDataset, get_transforms
from library.models import get_model
from library.utils import seed_everything, calculate_metric, get_class_weights


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Convert one-hot encoded labels to class indices for CrossEntropyLoss
        targets = torch.argmax(labels, dim=1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to validate on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Convert one-hot encoded labels to class indices for CrossEntropyLoss
            targets = torch.argmax(labels, dim=1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply softmax to get probabilities for ROC AUC calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate ROC AUC
    metric = calculate_metric(all_labels, all_preds)

    return epoch_loss, metric


def run_training_fold(model_name, train_df, valid_df, fold_idx):
    """
    Runs the training pipeline for a single fold.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet34').
        train_df (pd.DataFrame): DataFrame containing training data for this fold.
        valid_df (pd.DataFrame): DataFrame containing validation data for this fold.
        fold_idx (int): The index of the current fold (0-based).

    Returns:
        float: The best validation ROC AUC score achieved.
    """
    # Ensure output directory exists
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Set seed for reproducibility unique to this fold
    seed_everything(Config.SEED + fold_idx)

    device = Config.DEVICE

    # Prepare Datasets
    train_dataset = AppleLeafDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    valid_dataset = AppleLeafDataset(
        valid_df, transforms=get_transforms("valid"), mode="train"
    )

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = get_model(model_name, Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    # Calculate Class Weights for Loss
    class_weights = get_class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    # T_0 is set to EPOCHS in Config, implying one cycle over the full training duration
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
    )

    best_metric = -np.inf
    best_model_path = os.path.join(models_dir, f"{model_name}_fold_{fold_idx}.pth")

    print(f"Starting training for {model_name} - Fold {fold_idx}")
    print(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}"
    )

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, valid_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_metric:.15f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save best model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! AUC: {best_metric:.15f}")

    print(f"Fold {fold_idx} finished. Best AUC: {best_metric:.15f}")

    # Cleanup
    del model, optimizer, scheduler, train_loader, valid_loader
    torch.cuda.empty_cache()

    return best_metric
