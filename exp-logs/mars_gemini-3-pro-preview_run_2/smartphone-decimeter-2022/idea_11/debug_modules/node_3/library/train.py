import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import IMULocalTrajectoryCNN
from library.data_loader import load_data


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Average validation loss.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def train_model(debug_sample_fraction=None, load_cached_data=True):
    """
    Main function to train the model.

    Args:
        debug_sample_fraction: Float (0.0-1.0) to sample a fraction of training data for debugging.
                               If None, uses full dataset (or Config.DEBUG_SAMPLE_SIZE if set).
        load_cached_data: Boolean, whether to load pre-processed data from cache.
    """
    set_seed(Config.RANDOM_STATE)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    # If debug_sample_fraction is provided via argument, it overrides Config
    sample_frac = (
        debug_sample_fraction
        if debug_sample_fraction is not None
        else Config.DEBUG_SAMPLE_SIZE
    )

    train_dataset, val_dataset = load_data(
        mode="train", load_cached_data=load_cached_data, sample_fraction=sample_frac
    )

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 2. Initialize Model
    model = IMULocalTrajectoryCNN(
        input_dim=Config.INPUT_DIM,
        window_size=Config.WINDOW_SIZE,
        output_dim=Config.OUTPUT_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        cnn_dropout=Config.CNN_DROPOUT,
        mlp_hidden_dims=Config.MLP_HIDDEN_DIMS,
        mlp_dropout=Config.MLP_DROPOUT,
    )
    model = model.to(device)

    # 3. Setup Training Components
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler to reduce LR when validation loss plateaus
    # verbose argument removed as it causes TypeError in PyTorch 2.2+ (Cite debug_lesson_1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.8f} to {val_loss:.8f}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    print(f"Best Validation Loss: {best_val_loss:.8f}")
