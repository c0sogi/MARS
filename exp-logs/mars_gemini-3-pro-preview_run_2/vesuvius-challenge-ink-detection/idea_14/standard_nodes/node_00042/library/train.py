import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, save_checkpoint, f05_score
from library.dataset import InkDataset
from library.model import SiameseSegFormer
from library.losses import BCEDiceLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move data to device
        view_1 = inputs["view_1"].to(device)
        view_2 = inputs["view_2"].to(device)
        view_3 = inputs["view_3"].to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(view_1, view_2, view_3)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates Global F0.5 Score by accumulating TP, FP, FN over the dataset.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global F0.5
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for inputs, targets in loader:
            view_1 = inputs["view_1"].to(device)
            view_2 = inputs["view_2"].to(device)
            view_3 = inputs["view_3"].to(device)
            targets = targets.to(device)

            outputs = model(view_1, view_2, view_3)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Metrics calculation
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Calculate TP, FP, FN for this batch
            tp = (preds * targets).sum().item()
            fp = (preds * (1 - targets)).sum().item()
            fn = ((1 - preds) * targets).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = running_loss / len(loader)

    # Calculate Global F0.5
    beta = 0.5
    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)

    score = ((1 + beta**2) * precision * recall) / (
        beta**2 * precision + recall + epsilon
    )

    return avg_loss, score


def train():
    """
    Main training routine.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("Loading Metadata...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VALID_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata is generated."
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALID_METADATA_PATH)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create Datasets
    # load_cached_data=True allows using pre-processed .npy files if available
    train_dataset = InkDataset(train_df, mode="train", load_cached_data=True)
    val_dataset = InkDataset(val_df, mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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
        drop_last=False,
    )

    # 3. Model Initialization
    print(f"Initializing SiameseSegFormer (Backbone: {Config.MODEL_BACKBONE})...")
    model = SiameseSegFormer(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    )
    model.to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )
    criterion = BCEDiceLoss()

    # 5. Training Loop
    best_score = 0.0
    early_stopping_counter = 0

    print("Starting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F0.5: {val_score}"
        )

        # Validation Gating & Checkpointing
        # Only save if score > baseline AND score > current best
        if val_score > Config.BASELINE_SCORE:
            if val_score > best_score + Config.EARLY_STOPPING_MIN_DELTA:
                print(
                    f"New best score ({val_score} > {best_score}). Saving checkpoint..."
                )
                best_score = val_score
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    Config.CHECKPOINT_PATH,
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
        else:
            print(f"Score {val_score} did not exceed baseline {Config.BASELINE_SCORE}.")
            early_stopping_counter += 1

        # Early Stopping
        if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation F0.5 Score: {best_score}")
