import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import HCCRDSModel
from library.data_loader import process_data


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


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        atomic_features = batch["atomic_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        global_features = batch["global_features"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_features, batch_indices, global_features)

        # Compute loss (MSE on log-transformed targets)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

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
            # Move data to device
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(atomic_features, batch_indices, global_features)

            # Compute loss
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def run_training(sample_size=None):
    """
    Main training loop with Early Stopping and Learning Rate Scheduling.

    Args:
        sample_size (int, optional): Number of samples to use for debugging.

    Returns:
        model (nn.Module): The trained model with best validation weights.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load data
    print("Initializing data loaders...")
    train_loader, val_loader, test_loader = process_data(
        load_cached_data=True, sample_size=sample_size
    )

    # Initialize model
    print("Initializing HC-CRDS model...")
    model = HCCRDSModel().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Loss function (MSE for regression)
    criterion = nn.MSELoss()

    # Training Loop variables
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Loss: {val_loss:.8f} - "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"  -> New best model saved (Val Loss: {val_loss:.8f})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Val Loss: {best_val_loss:.8f}")

    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT))

    return model


def generate_submission(model, device=None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (nn.Module): Trained model.
        device (torch.device, optional): Device to run inference on.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    print("Generating submission...")

    # Get test loader
    # We rely on process_data to return the loader consistent with training
    _, _, test_loader = process_data(load_cached_data=True)

    model.eval()
    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            outputs = model(atomic_features, batch_indices, global_features)

            # Inverse transform targets: exp(y) - 1
            # Since we trained on log1p(y)
            preds_original_scale = torch.expm1(outputs)

            ids.extend(batch_ids.tolist())
            predictions.extend(preds_original_scale.cpu().numpy())

    # Create DataFrame
    predictions = np.array(predictions)
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to ensure correct order
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission_df.head())
