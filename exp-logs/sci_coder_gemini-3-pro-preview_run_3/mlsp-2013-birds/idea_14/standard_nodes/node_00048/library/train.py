import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    seed_everything,
    ModelEMA,
    calculate_roc_auc,
    save_checkpoint,
)
from library.dataset import (
    BirdDataset,
    get_transforms,
    get_data_splits,
    mixup_data,
)
from library.models import BirdClassifier


def train_one_epoch(train_loader, model, ema, optimizer, criterion, device):
    """
    Executes one training epoch with Mixup and EMA updates.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        # alpha=0.4 is a common default, could be parameterized but fixed here for simplicity
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, alpha=0.4, use_cuda=True
        )

        # Forward pass
        optimizer.zero_grad()
        logits = model(mixed_images)

        # Mixup Loss
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(
            logits, labels_b
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        if ema:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set using robust AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        # Calculate robust AUC
        epoch_auc = calculate_roc_auc(all_targets, all_preds)
    else:
        epoch_auc = 0.0

    return epoch_loss, epoch_auc


import numpy as np  # Needed for concatenation in validate


def run_fold(fold_idx, backbone, data_source):
    """
    Runs the training and validation loop for a specific fold, backbone, and data source.
    Saves the best EMA model checkpoint.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define checkpoint name
    ckpt_name = f"{backbone}_{data_source}_fold_{fold_idx}.pth"

    print(f"Starting run: Fold {fold_idx} | Backbone {backbone} | Source {data_source}")

    # 2. Data Preparation
    # Load folds dataframe (cached)
    df = get_data_splits(load_cached_data=True)

    # Split into train and val
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)

    # Create Datasets
    train_dataset = BirdDataset(
        train_df,
        data_source=data_source,
        phase="train",
        transform=get_transforms("train"),
    )
    val_dataset = BirdDataset(
        val_df, data_source=data_source, phase="val", transform=get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last to avoid issues with batch norm on small leftover batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = BirdClassifier(backbone_name=backbone, pretrained=True)
    model.to(device)

    # Initialize EMA
    # Decay is set in Config (e.g., 0.95 for small data)
    ema = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    early_stop_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, ema, optimizer, criterion, device
        )

        # Validate (using EMA model as it is the target for inference)
        val_loss, val_auc = validate(val_loader, ema.module, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            early_stop_counter = 0

            # Save the EMA model state as the best model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": ema.module.state_dict(),  # Saving EMA weights
                    "best_score": float(best_auc),
                    "optimizer": optimizer.state_dict(),
                    "config": {
                        "backbone": backbone,
                        "data_source": data_source,
                        "fold": fold_idx,
                    },
                },
                is_best=True,
                filename=ckpt_name,
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Fold {fold_idx} finished. Best Val AUC: {best_auc:.10f}")

    # Clean up
    del model, ema, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_auc
