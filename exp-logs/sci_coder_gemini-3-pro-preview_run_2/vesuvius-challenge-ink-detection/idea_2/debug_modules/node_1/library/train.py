import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed
from library.dataset import InkDataset
from library.model import InkDetector
from library.losses import BCEDiceLoss
from library.metrics import calculate_fbeta


def train_model(load_cached_data=True):
    """
    Executes the training pipeline for the Vesuvius Ink Detection model.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed fragment volumes
                                 from disk to speed up initialization.
    """
    # 1. Setup and Seeding
    Config.setup()
    set_seed(Config.SEED)

    print(f"Initializing training with Experiment Name: {Config.EXP_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 2. Load Metadata
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(Config.METADATA_DIR, "validation.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            "Metadata CSV files not found. Ensure metadata generation script has run."
        )

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Debug Mode: Subsample data
    if Config.DEBUG:
        print("DEBUG mode enabled: Subsampling datasets.")
        train_df = train_df.sample(
            n=min(len(train_df), 32), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), 16), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Initialize Datasets and Loaders
    # Using InkDataset from library.dataset which handles Albumentations and Caching
    train_dataset = InkDataset(
        train_df, mode="train", load_cached_data=load_cached_data
    )
    val_dataset = InkDataset(
        val_df, mode="validation", load_cached_data=load_cached_data
    )

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

    # 4. Initialize Model, Loss, Optimizer
    model = InkDetector()
    model.to(Config.DEVICE)

    criterion = BCEDiceLoss()
    optimizer = Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training Loop
    best_val_score = -1.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0

        for images, labels, masks in train_loader:
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = (
            train_loss_sum / len(train_loader) if len(train_loader) > 0 else 0.0
        )

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        val_score_sum = 0.0

        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)

                # Calculate F0.5 Score
                score = calculate_fbeta(
                    logits, labels, beta=0.5, threshold=Config.THRESHOLD
                )

                val_loss_sum += loss.item()
                val_score_sum += score

        avg_val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_score = val_score_sum / len(val_loader) if len(val_loader) > 0 else 0.0

        # --- Logging ---
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val F0.5: {avg_val_score}"
        )

        # --- Scheduler Step ---
        scheduler.step(avg_val_loss)

        # --- Model Checkpointing ---
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! Score: {best_val_score}")
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Early Stopping ---
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Score: {best_val_score}"
            )
            break

    print("Training pipeline completed.")
