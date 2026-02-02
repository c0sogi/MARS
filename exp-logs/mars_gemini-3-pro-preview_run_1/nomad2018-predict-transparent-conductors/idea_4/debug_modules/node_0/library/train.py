import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_datasets, collate_fn
from library.model import APDeepSets


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(debug=False, max_samples=None, num_epochs=None):
    """
    Main training function.

    Args:
        debug (bool): If True, runs with a smaller subset of data.
        max_samples (int): Number of samples to use in debug mode.
        num_epochs (int): Number of epochs to train. If None, uses Config.NUM_EPOCHS.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = get_datasets(debug=debug, max_samples=max_samples)

    # Create DataLoaders
    # Note: Using custom collate_fn to handle variable size atomic features
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Initialize Model
    model = APDeepSets()
    model.to(device)

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training Loop Configuration
    epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move batch data to device
            # batch is a dict: {'ids', 'global_features', 'atomic_features', 'batch_indices', 'targets'}
            global_features = batch["global_features"].to(device)
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            # Prepare input for model
            model_input = {
                "global_features": global_features,
                "atomic_features": atomic_features,
                "batch_indices": batch_indices,
            }

            # Forward pass
            optimizer.zero_grad()
            outputs = model(model_input)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                global_features = batch["global_features"].to(device)
                atomic_features = batch["atomic_features"].to(device)
                batch_indices = batch["batch_indices"].to(device)
                targets = batch["targets"].to(device)

                model_input = {
                    "global_features": global_features,
                    "atomic_features": atomic_features,
                    "batch_indices": batch_indices,
                }

                outputs = model(model_input)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

        avg_val_loss = val_loss / len(val_dataset)

        # Print metrics (full precision)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
        )

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Early Stopping and Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")
