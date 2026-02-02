import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.models import SliceEncoder
from library.datasets import SliceClassificationDataset
from library.losses import EncoderLoss


class TrainableSliceEncoder(nn.Module):
    """
    Wrapper around SliceEncoder to add a classification head for training.
    The SliceEncoder outputs features; this adds a Linear layer to project to a logit.
    """

    def __init__(self, backbone_name, pretrained=True):
        super(TrainableSliceEncoder, self).__init__()
        self.encoder = SliceEncoder(backbone_name=backbone_name, pretrained=pretrained)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits


def train_encoder(debug=False):
    """
    Trains the Stage 2 Slice Encoder (2.5D CNN) on slice-level fracture labels.

    Args:
        debug (bool): If True, runs on a small subset of data for testing.
    """
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Starting Encoder Training on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        print("Debug mode: limiting metadata size.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Prepare Datasets and Loaders
    # We use load_cached_data=False to ensure the dataset logic correctly filters
    # based on the provided metadata dataframe (train vs val) without cache collisions.
    print("Initializing Training Dataset...")
    train_dataset = SliceClassificationDataset(
        metadata_df=train_df, load_cached_data=False
    )

    print("Initializing Validation Dataset...")
    val_dataset = SliceClassificationDataset(metadata_df=val_df, load_cached_data=False)

    if len(train_dataset) == 0:
        print(
            "Warning: No training samples with bounding boxes found. Skipping training."
        )
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.ENCODER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.ENCODER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Initialize Model, Loss, Optimizer
    # We wrap the encoder to add a classification head
    model = TrainableSliceEncoder(
        backbone_name=Config.ENCODER_BACKBONE, pretrained=True
    )
    model.to(device)

    criterion = EncoderLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.ENCODER_LR)

    # 5. Training Loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    epochs = Config.ENCODER_EPOCHS if not debug else 2

    print("Starting training loop...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Match logit shape (Batch, 1)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

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
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save only the encoder part (without the temporary classification head)
            # This allows Stage 3 to load it as a pure feature extractor
            save_path = os.path.join(Config.CHECKPOINT_DIR, "slice_encoder.pth")
            torch.save(model.encoder.state_dict(), save_path)
            print(f"Validation loss improved. Encoder model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Encoder training completed.")
