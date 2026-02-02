import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SEED,
    BATCH_SIZE,
)
from library.utils import seed_everything, calculate_mae
from library.model_cnn import SeismicCNN
from library.data_processing import get_spectrogram_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The CNN model.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        float: Average training loss (MAE) for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Match shape (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs one epoch of validation.

    Args:
        model (nn.Module): The CNN model.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run on.

    Returns:
        float: Average validation loss (MAE) for the epoch.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

    return running_loss / count


def run_cnn_training(debug=False):
    """
    Orchestrates the full training cycle for the CNN model.
    Includes data loading, model initialization, training loop, early stopping,
    and saving the best model.

    Args:
        debug (bool): If True, runs on a small subset of data.

    Returns:
        model (nn.Module): The trained model with best weights loaded.
        val_mae (float): The best validation MAE achieved.
    """
    seed_everything(SEED)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Starting CNN Training on device: {device}")

    # 1. Prepare Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_spectrogram_loaders(
        batch_size=BATCH_SIZE, debug=debug
    )

    # 2. Initialize Model
    model = SeismicCNN().to(device)

    # 3. Setup Training Components
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 4. Training Loop
    best_val_mae = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "cnn_best.pth")

    print(f"Training for {EPOCHS} epochs with patience {PATIENCE}...")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{EPOCHS} - Train MAE: {train_loss} - Val MAE: {val_loss}")

        # Early Stopping & Model Saving
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Load Best Weights
    print(f"Loading best weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_val_mae


def predict_cnn(model, test_loader=None, device=None):
    """
    Generates predictions for the test set using the CNN model.

    Args:
        model (nn.Module): Trained CNN model.
        test_loader (DataLoader, optional): If None, creates a new one.
        device (torch.device, optional): Device to run on.

    Returns:
        pd.DataFrame: DataFrame with 'segment_id' and 'time_to_eruption'.
    """
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    if test_loader is None:
        _, _, test_loader = get_spectrogram_loaders(batch_size=BATCH_SIZE, debug=False)

    model.eval()
    model.to(device)

    segment_ids = []
    predictions = []

    print("Generating CNN predictions...")
    with torch.no_grad():
        for inputs, _, ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Collect results
            # outputs shape is (B, 1), flatten to (B,)
            preds = outputs.squeeze(1).cpu().numpy()

            predictions.extend(preds)
            segment_ids.extend(ids.numpy())

    # Create DataFrame
    results_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure segment_id is int
    results_df["segment_id"] = results_df["segment_id"].astype(int)

    return results_df
