import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import time

from library.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    NUM_WORKERS,
    SEED,
    WORK_DIR,
)
from library.data import load_dataset
from library.model import ResidualMLP


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
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
    Executes one training epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def validate_epoch(model, dataloader, criterion, device):
    """
    Executes validation for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def run_training(
    load_cached_data=True,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    patience=EARLY_STOPPING_PATIENCE,
):
    """
    Orchestrates the entire training pipeline: data loading, model initialization,
    training loops, and early stopping.

    Args:
        load_cached_data (bool): Whether to attempt loading data from parquet cache.
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        nn.Module: The trained model with the best validation weights.
    """
    set_seed(SEED)

    # Ensure working directory exists for model checkpoint
    os.makedirs(WORK_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = load_dataset(mode="train", load_cached_data=load_cached_data)
    val_dataset = load_dataset(mode="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 2. Initialize Model
    # Determine input dimension dynamically from the dataset
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]

    model = ResidualMLP(input_dim=input_dim).to(device)

    # 3. Setup Training Components
    criterion = nn.L1Loss()  # Mean Absolute Error is robust for GNSS residuals
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Validation loss improved. Model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s.")
    print(f"Best Validation Loss: {best_val_loss}")

    # Load best model state
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    return model
