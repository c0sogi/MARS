import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import DualShellWideDeepSets


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_atom_feats, batch_indices, batch_global_feats, batch_targets, _ in loader:
        # Move data to device
        batch_atom_feats = batch_atom_feats.to(device)
        batch_indices = batch_indices.to(device)
        batch_global_feats = batch_global_feats.to(device)
        batch_targets = batch_targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(batch_atom_feats, batch_indices, batch_global_feats)

        # Compute loss
        loss = criterion(preds, batch_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        batch_size = batch_targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (
            batch_atom_feats,
            batch_indices,
            batch_global_feats,
            batch_targets,
            _,
        ) in loader:
            # Move data to device
            batch_atom_feats = batch_atom_feats.to(device)
            batch_indices = batch_indices.to(device)
            batch_global_feats = batch_global_feats.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            preds = model(batch_atom_feats, batch_indices, batch_global_feats)

            # Compute loss
            loss = criterion(preds, batch_targets)

            # Accumulate loss
            batch_size = batch_targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / total_samples


def run_training(train_loader, val_loader):
    """
    Orchestrates the training process with early stopping and LR scheduling.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = DualShellWideDeepSets().to(device)

    # Initialize Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.8f}")

    # Load best model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    return model


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    all_ids = []
    all_preds = []

    print("Generating submission...")

    with torch.no_grad():
        for (
            batch_atom_feats,
            batch_indices,
            batch_global_feats,
            _,
            batch_ids,
        ) in test_loader:
            # Move data to device
            batch_atom_feats = batch_atom_feats.to(device)
            batch_indices = batch_indices.to(device)
            batch_global_feats = batch_global_feats.to(device)

            # Forward pass (predictions are in log scale)
            preds_log = model(batch_atom_feats, batch_indices, batch_global_feats)

            # Inverse transform: exp(x) - 1
            # Clamp to avoid numerical instability if necessary, though expm1 handles small x well.
            preds_original = torch.expm1(preds_log)

            # Collect results
            all_ids.append(batch_ids.cpu().numpy())
            all_preds.append(preds_original.cpu().numpy())

    # Concatenate all batches
    all_ids = np.concatenate(all_ids)
    all_preds = np.concatenate(all_preds, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID to ensure consistent order
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
