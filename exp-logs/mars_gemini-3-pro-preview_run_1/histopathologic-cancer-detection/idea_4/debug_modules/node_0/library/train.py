import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, save_checkpoint, MetricTracker
from library.dataset import TumorDataset
from library.model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    tracker = MetricTracker()

    for images, labels in loader:
        images = images.to(device)
        # Ensure labels are (N, 1) for BCEWithLogitsLoss
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Apply sigmoid to get probabilities for AUC calculation
        preds = torch.sigmoid(outputs)
        tracker.update(loss.item(), preds, labels)

    return tracker.get_avg_loss(), tracker.get_auc()


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.sigmoid(outputs)
            tracker.update(loss.item(), preds, labels)

    return tracker.get_avg_loss(), tracker.get_auc()


def run_training():
    """
    Main training execution function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Loading metadata from {Config.METADATA_DIR}...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Debug mode subsampling
    if Config.DEBUG:
        print(f"DEBUG mode enabled. Subsampling {Config.DEBUG_SAMPLE_SIZE} records.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. DataLoaders
    train_dataset = TumorDataset(train_df, split="train")
    val_dataset = TumorDataset(val_df, split="val")

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

    # 3. Model Setup
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = get_model()
    model = model.to(Config.DEVICE)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Train AUC:  {train_auc}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val AUC:    {val_auc}")

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filepath=Config.CHECKPOINT_PATH,
            )
            print(f"  [SAVED] New best model with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"  [INFO] No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
