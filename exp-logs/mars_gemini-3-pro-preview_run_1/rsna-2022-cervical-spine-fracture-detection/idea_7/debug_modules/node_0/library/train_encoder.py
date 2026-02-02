import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.models import FractureEncoder
from library.data import SliceClassificationDataset


class FractureClassifier(nn.Module):
    """
    Wrapper class to add a classification head to the FractureEncoder
    for Stage 2 training (Slice/Crop Classification).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Input is the embedding dimension from the encoder
        self.head = nn.Linear(Config.STAGE2_EMBEDDING_DIM, 1)

    def forward(self, x):
        # Get embedding from encoder
        embedding = self.encoder(x)
        # Project to logit
        logit = self.head(embedding)
        return logit


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()

    return running_loss / len(loader)


def train_stage2(load_cached_data=True):
    """
    Main function to train the Stage 2 Fracture Encoder.
    Trains on slice-level binary classification (Fracture vs No Fracture).
    Saves the encoder weights (without the classification head) for Stage 3.

    Args:
        load_cached_data (bool): Whether to use cached dataset indices/files.
    """
    print("Initializing Stage 2: Fracture Encoder Training...")

    # 1. Setup Device and Config
    device = torch.device(Config.DEVICE)
    Config.setup()

    # 2. Prepare Datasets and Loaders
    # Note: SliceClassificationDataset handles balancing and crop extraction
    train_dataset = SliceClassificationDataset(
        split="train", load_cached_data=load_cached_data
    )
    val_dataset = SliceClassificationDataset(
        split="val", load_cached_data=load_cached_data
    )

    if len(train_dataset) == 0:
        print("No training data found for slice classification. Exiting Stage 2.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.STAGE2_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.STAGE2_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model Setup
    # Initialize the base encoder
    encoder = FractureEncoder()
    # Wrap it for classification training
    model = FractureClassifier(encoder).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.STAGE2_LR, weight_decay=Config.STAGE2_WEIGHT_DECAY
    )

    # Unweighted BCE as specified to focus on local precision
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")

    print(f"Starting training for {Config.STAGE2_EPOCHS} epochs...")

    for epoch in range(Config.STAGE2_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.STAGE2_EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save ONLY the encoder part, not the temporary classification head
            # This allows Stage 3 to load just the feature extractor
            torch.save(model.encoder.state_dict(), checkpoint_path)
            print(f"  New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Stage 2 Training Completed.")
