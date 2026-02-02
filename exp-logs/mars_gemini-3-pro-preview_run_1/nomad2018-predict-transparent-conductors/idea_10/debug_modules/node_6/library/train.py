import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, rmsle
from library.data import get_dataloaders
from library.model import RTDSModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for atom_x, glob_x, targets, batch_indices, _ in loader:
        atom_x = atom_x.to(device)
        glob_x = glob_x.to(device)
        targets = targets.to(device)
        batch_indices = batch_indices.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(atom_x, glob_x, batch_indices)

        # Loss calculation (MSE on log-transformed targets)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Validation loop. Calculates MSE Loss and RMSLE metric.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for atom_x, glob_x, targets, batch_indices, _ in loader:
            atom_x = atom_x.to(device)
            glob_x = glob_x.to(device)
            targets = targets.to(device)
            batch_indices = batch_indices.to(device)

            outputs = model(atom_x, glob_x, batch_indices)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

            # Store predictions and targets for metric calculation
            # Inverse transform from log1p domain to original domain for RMSLE calculation
            # The dataset returns log1p(target), model predicts log1p(target)
            preds_original = torch.expm1(outputs)
            targets_original = torch.expm1(targets)

            all_preds.append(preds_original.cpu().numpy())
            all_targets.append(targets_original.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate RMSLE using the utility function
    # Note: rmsle function applies log1p internally, so we pass original scale values
    metric = rmsle(all_targets, all_preds)

    return epoch_loss, metric


def run_training(load_cached_data=True, debug_size=None):
    """
    Main training execution function.
    """
    # Set reproducibility
    seed_everything(Config.SEED)

    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    print("Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = RTDSModel().to(device)

    # Loss function: MSE is appropriate because targets are log-transformed
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.6f}")


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.
    """
    print("Generating submission...")
    seed_everything(Config.SEED)

    # Get test loader
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug_size=None
    )

    device = torch.device(Config.DEVICE)
    model = RTDSModel().to(device)

    # Load best model weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_ids = []
    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for atom_x, glob_x, _, batch_indices, ids in test_loader:
            atom_x = atom_x.to(device)
            glob_x = glob_x.to(device)
            batch_indices = batch_indices.to(device)

            # Forward pass
            outputs = model(atom_x, glob_x, batch_indices)

            # Inverse transform: exp(x) - 1
            # We clip negative predictions to 0 just in case, though expm1 handles it naturally
            preds_original = torch.expm1(outputs).cpu().numpy()

            all_ids.extend(ids)
            all_preds.append(preds_original)

    all_preds = np.concatenate(all_preds, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
