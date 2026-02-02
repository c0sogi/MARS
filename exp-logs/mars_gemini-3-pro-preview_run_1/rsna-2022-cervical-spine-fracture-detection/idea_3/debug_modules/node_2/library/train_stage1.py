import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.models import SpineLocalizer
from library.datasets import SegmentationDataset
from library.losses import DiceLoss


def train_localizer(debug=False):
    """
    Trains the Stage 1 Spine Localizer (U-Net).

    Args:
        debug (bool): If True, runs on a small subset of data for testing.
    """
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Starting Localizer Training on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        print("Debug mode: limiting metadata size.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Prepare Datasets and Loaders
    # Note: SegmentationDataset filters internally for studies with 'has_segmentation'==True
    print("Initializing Training Dataset...")
    train_dataset = SegmentationDataset(metadata_df=train_df, load_cached_data=True)

    print("Initializing Validation Dataset...")
    val_dataset = SegmentationDataset(metadata_df=val_df, load_cached_data=True)

    if len(train_dataset) == 0:
        print(
            "Warning: No training samples with segmentation found. Skipping training."
        )
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.LOCALIZER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.LOCALIZER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Initialize Model, Loss, Optimizer
    model = SpineLocalizer(pretrained=True)
    model.to(device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LOCALIZER_LR)

    # 5. Training Loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    epochs = Config.LOCALIZER_EPOCHS if not debug else 2

    print("Starting training loop...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Dice Loss: {avg_train_loss} - Val Dice Loss: {avg_val_loss}"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            save_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Localizer training completed.")
