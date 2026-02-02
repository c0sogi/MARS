import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time

import library.config as config
from library.model import CCWDS
from library.data_loader import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        # batch_atomic: (Sum_N, 12)
        # batch_indices: (Sum_N,)
        # batch_global: (B, 12)
        # batch_targets: (B, 2)
        # batch_ids: (B,)
        batch_atomic, batch_indices, batch_global, batch_targets, _ = batch

        # Move to device
        batch_atomic = batch_atomic.to(device)
        batch_indices = batch_indices.to(device)
        batch_global = batch_global.to(device)
        batch_targets = batch_targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_atomic, batch_indices, batch_global)

        # Compute loss
        loss = criterion(outputs, batch_targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * batch_targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch_atomic, batch_indices, batch_global, batch_targets, _ = batch

            batch_atomic = batch_atomic.to(device)
            batch_indices = batch_indices.to(device)
            batch_global = batch_global.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_atomic, batch_indices, batch_global)

            loss = criterion(outputs, batch_targets)
            running_loss += loss.item() * batch_targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    # Since targets are log1p transformed, MSE loss corresponds to Mean Squared Logarithmic Error.
    # RMSLE is sqrt(MSE).
    rmsle = np.sqrt(epoch_loss)

    return epoch_loss, rmsle


def train_model(
    epochs=None,
    patience=None,
    learning_rate=None,
    weight_decay=None,
    batch_size=None,
    max_samples=None,
):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # Cite debug_lesson_14: Default Arguments Bind at Definition Time
    if epochs is None:
        epochs = config.EPOCHS
    if patience is None:
        patience = config.PATIENCE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    config.setup_directories()
    config.set_seed()

    print(f"Device: {config.DEVICE}")

    # Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=True, max_samples=max_samples
    )

    # Initialize Model
    model = CCWDS().to(config.DEVICE)

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler
    # Cite debug_lesson_2: Remove Deprecated verbose Argument
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR,
    )

    # Training Loop variables
    best_val_loss = float("inf")
    best_val_rmsle = float("inf")
    epochs_no_improve = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )

        # Validate
        val_loss, val_rmsle = evaluate(model, val_loader, criterion, config.DEVICE)

        # Update Scheduler
        scheduler.step(val_loss)

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmsle = val_rmsle
            epochs_no_improve = 0
            # Cite debug_lesson_16: Ensure directory exists
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            saved_str = " [Saved Best Model]"
        else:
            epochs_no_improve += 1
            saved_str = ""

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val Loss (MSE): {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle:.6f} | "
            f"Time: {elapsed:.2f}s{saved_str}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val RMSLE: {best_val_rmsle:.6f}")

    # Load best model for return
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    return model, test_loader


def generate_submission(model, test_loader, device, save_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            batch_atomic, batch_indices, batch_global, _, batch_ids = batch

            batch_atomic = batch_atomic.to(device)
            batch_indices = batch_indices.to(device)
            batch_global = batch_global.to(device)

            outputs = model(batch_atomic, batch_indices, batch_global)

            # Inverse transform: exp(y) - 1
            # Since targets were log1p(y)
            preds = torch.expm1(outputs)

            # Clamp predictions to be non-negative (physics constraint)
            preds = torch.clamp(preds, min=0.0)

            ids_list.extend(batch_ids.numpy())
            preds_list.extend(preds.cpu().numpy())

    # Create DataFrame
    preds_array = np.array(preds_list)
    submission_df = pd.DataFrame(
        {
            "id": ids_list,
            "formation_energy_ev_natom": preds_array[:, 0],
            "bandgap_energy_ev": preds_array[:, 1],
        }
    )

    # Sort by ID just in case
    submission_df.sort_values("id", inplace=True)

    # Save
    # Cite debug_lesson_16: Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training(max_samples=None, epochs=None):
    """
    Main entry point to run the full training and submission pipeline.
    """
    if epochs is None:
        epochs = config.EPOCHS
    # Train the model
    best_model, test_loader = train_model(epochs=epochs, max_samples=max_samples)

    # Generate submission
    generate_submission(best_model, test_loader, config.DEVICE, config.SUBMISSION_PATH)
