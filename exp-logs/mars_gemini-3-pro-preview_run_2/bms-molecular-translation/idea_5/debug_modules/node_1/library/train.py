import os
import gc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ChemicalDataset
from library.model import StoichiometryEncoder


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # The model returns (embedding, atom_preds). We only need atom_preds for training.
        _, preds = model(images)

        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            _, preds = model(images)
            loss = criterion(preds, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_training(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Main driver function to set up data, model, and run the training loop.

    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of training epochs.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Loading metadata from {Config.METADATA_DIR}...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        print(f"Debug mode enabled. Using {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Initialize Datasets
    train_dataset = ChemicalDataset(df_train, mode="train")
    val_dataset = ChemicalDataset(df_val, mode="val")

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    device = Config.DEVICE
    print(f"Initializing StoichiometryEncoder on {device}...")
    model = StoichiometryEncoder(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        embedding_dim=Config.EMBEDDING_DIM,
        num_atoms=Config.NUM_ATOMS,
    )
    model = model.to(device)

    # Setup Loss, Optimizer, Scheduler
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # Scheduler Step
        scheduler.step(val_loss)

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"Validation loss improved. Model saved to {Config.MODEL_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("-" * 30)

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    print("Training complete.")
