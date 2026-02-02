import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import prepare_datasets, VentilatorDataset
from library.model import CWDHNet, MaskedMAELoss


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(inputs)
        # MaskedMAELoss requires inputs to identify the inspiratory phase (u_out == 0)
        loss = criterion(preds, targets, inputs)

        loss.backward()

        # Gradient Clipping to stabilize hybrid architecture
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets, inputs)

            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Orchestrates the training process with Early Stopping and Scheduler.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load datasets (cached or fresh)
    train_x, train_y, val_x, val_y, _ = prepare_datasets(load_cached_data=True)

    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model, Criterion, Optimizer
    model = CWDHNet().to(device)
    criterion = MaskedMAELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler based on Config parameters (Plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
        min_lr=Config.MIN_LR,
    )

    best_loss = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! Loss: {best_loss}")
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Loss: {best_loss}")


def generate_submission(batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set and saves the submission file.
    """
    set_seed()
    device = torch.device(Config.DEVICE)

    # Load test data (ignore train/val returns)
    _, _, _, _, test_x = prepare_datasets(load_cached_data=True)

    # Load test_ids from cache (created by prepare_datasets)
    test_ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(
            "test_ids.npy not found. Run training/preparation first."
        )
    test_ids = np.load(test_ids_path)

    test_dataset = VentilatorDataset(test_x, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = CWDHNet().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            predictions.append(preds.cpu().numpy().flatten())

    predictions = np.concatenate(predictions)
    flat_ids = test_ids.flatten()

    # Ensure lengths match (safety check)
    if len(flat_ids) != len(predictions):
        print(
            f"Warning: Length mismatch! IDs: {len(flat_ids)}, Preds: {len(predictions)}"
        )
        min_len = min(len(flat_ids), len(predictions))
        flat_ids = flat_ids[:min_len]
        predictions = predictions[:min_len]

    submission_df = pd.DataFrame({"id": flat_ids, "pressure": predictions})

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
