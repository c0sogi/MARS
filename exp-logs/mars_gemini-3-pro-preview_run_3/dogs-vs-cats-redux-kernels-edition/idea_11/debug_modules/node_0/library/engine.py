import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.utils import AverageMeter, save_checkpoint, get_device, ensure_dir
from library.models import CustomEnsembleModel
from library.dataset import DogCatDataset
from library.transforms import get_transforms
from library.config import WORKING_DIR, DataConfig, TrainConfig, ModelConfig


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # Output shape: (Batch_Size, 1)
        logits = model(images)

        # Ensure labels match logits shape
        labels = labels.view_as(logits)

        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            labels = labels.view_as(logits)

            loss = criterion(logits, labels)
            losses.update(loss.item(), images.size(0))

            # Calculate accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return losses.avg, accuracy


def train_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
):
    """
    Main function to train a single model configuration.

    Args:
        model_config: Configuration for the specific model architecture.
        data_config: Configuration for data paths.
        train_config: Configuration for training hyperparameters.

    Returns:
        float: The best validation loss achieved.
    """
    device = get_device()
    print(f"Training {model_config.model_name} on {device}")

    # 1. Load Data
    train_df = pd.read_csv(data_config.train_csv)
    val_df = pd.read_csv(data_config.val_csv)

    # 2. Prepare Transforms
    train_transform = get_transforms(model_config.input_size, mode="train")
    val_transform = get_transforms(model_config.input_size, mode="val")

    # 3. Create Datasets and Loaders
    train_dataset = DogCatDataset(train_df, transform=train_transform, mode="train")
    val_dataset = DogCatDataset(val_df, transform=val_transform, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = CustomEnsembleModel(
        config=model_config, num_classes=data_config.num_classes, pretrained=True
    )
    model = model.to(device)

    # 5. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_config.epochs, eta_min=train_config.min_lr
    )

    # 6. Loss Function
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 7. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    ensure_dir(WORKING_DIR)

    # Unique identifier for saving files for this specific model config
    # Clean model name to be path safe
    safe_model_name = model_config.model_name.replace(".", "_")
    checkpoint_dir = os.path.join(WORKING_DIR, safe_model_name)
    ensure_dir(checkpoint_dir)

    print(f"Starting training for {train_config.epochs} epochs...")

    for epoch in range(train_config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{train_config.epochs} - "
            f"Time: {elapsed}s - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val Acc: {val_acc}"
        )

        # Checkpointing
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            print(f"New best model found! Saving to {checkpoint_dir}")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "config": model_config,
            },
            is_best,
            checkpoint_dir,
        )

        # Early Stopping
        if patience_counter >= train_config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(
        f"Training finished for {model_config.model_name}. Best Val Loss: {best_loss}"
    )
    return best_loss
