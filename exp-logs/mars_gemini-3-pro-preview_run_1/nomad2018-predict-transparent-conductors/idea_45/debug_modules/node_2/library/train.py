import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.utils import set_seed, rmsle, expm1_transform
from library.data import get_loaders
from library.model import CEAMSDS


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        # Unpack batch from collate_sparse
        x_atomic, x_global, y, batch_indices = batch

        # Move to device
        x_atomic = x_atomic.to(device)
        x_global = x_global.to(device)
        y = y.to(device)
        batch_indices = batch_indices.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_atomic, x_global, batch_indices)

        # Compute loss
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes MSE Loss (log-space) and RMSLE (original space).
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_atomic, x_global, y, batch_indices = batch

            x_atomic = x_atomic.to(device)
            x_global = x_global.to(device)
            y = y.to(device)
            batch_indices = batch_indices.to(device)

            outputs = model(x_atomic, x_global, batch_indices)

            loss = criterion(outputs, y)
            running_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate predictions and targets
    y_pred_log = np.concatenate(all_preds, axis=0)
    y_true_log = np.concatenate(all_targets, axis=0)

    # Inverse transform to get back to original scale for RMSLE calculation
    # The targets were log1p transformed during loading, so we use expm1
    y_pred_orig = expm1_transform(y_pred_log)
    y_true_orig = expm1_transform(y_true_log)

    # Calculate RMSLE using the utility function
    # Note: library.utils.rmsle applies log1p internally, so we pass original scale values
    val_rmsle = rmsle(y_true_orig, y_pred_orig)

    return avg_loss, val_rmsle


def run_training(debug_size=None, num_epochs=Config.NUM_EPOCHS):
    """
    Main training loop with Early Stopping.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Get DataLoaders
    print("Loading data...")
    train_loader, val_loader, _ = get_loaders(
        batch_size=Config.BATCH_SIZE, debug_size=debug_size, load_cached_scalers=True
    )

    # Initialize Model
    model = CEAMSDS().to(device)

    # Optimizer and Loss
    # Using AdamW as specified
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    criterion = nn.MSELoss()

    # Early Stopping tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_PATH

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle} | "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss}")


def generate_submission(model=None, device=None):
    """
    Generates predictions for the test set using the best saved model.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    # Load data
    _, _, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_scalers=True
    )

    # Load model if not provided
    if model is None:
        model = CEAMSDS().to(device)
        if os.path.exists(Config.MODEL_PATH):
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
            print(f"Loaded model from {Config.MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            x_atomic, x_global, _, batch_indices = batch

            x_atomic = x_atomic.to(device)
            x_global = x_global.to(device)
            batch_indices = batch_indices.to(device)

            outputs = model(x_atomic, x_global, batch_indices)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate predictions
    y_pred_log = np.concatenate(all_preds, axis=0)

    # Inverse transform (log1p -> expm1)
    y_pred = expm1_transform(y_pred_log)

    # Clip negative values to 0 (physical energy constraint)
    y_pred = np.maximum(y_pred, 0)

    # Create submission DataFrame
    # We need the IDs from the test metadata
    test_df = pd.read_csv(Config.TEST_CSV)
    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "formation_energy_ev_natom": y_pred[:, 0],
            "bandgap_energy_ev": y_pred[:, 1],
        }
    )

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print head for verification
    print(submission_df.head())
