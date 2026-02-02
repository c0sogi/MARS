import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.data import get_dataloaders
from library.model import RA_CGN, train_one_epoch
from library.utils import set_seed


def compute_rmsle(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error.
    y_true, y_pred: numpy arrays of shape (N, 2)
    """
    # Ensure non-negative for log
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    squared_error = (log_pred - log_true) ** 2
    mean_squared_error = np.mean(squared_error, axis=0)
    rmsle_per_col = np.sqrt(mean_squared_error)

    return np.mean(rmsle_per_col)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler):
    """
    Evaluates the model on the given loader.
    Returns the average MSE loss (scaled) and the RMSLE metric (original scale).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        preds_scaled = model(batch)

        # Loss calculation (on scaled values, as per training objective)
        loss = criterion(preds_scaled, batch.y)
        total_loss += loss.item() * batch.num_graphs

        # Inverse transform for metric calculation
        preds_original = scaler.inverse_transform(preds_scaled.cpu().numpy())
        targets_original = scaler.inverse_transform(batch.y.cpu().numpy())

        all_preds.append(preds_original)
        all_targets.append(targets_original)

    avg_loss = total_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    rmsle_score = compute_rmsle(all_targets, all_preds)

    return avg_loss, rmsle_score


@torch.no_grad()
def generate_submission(model, loader, device, scaler, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []
    all_ids = []

    for batch in loader:
        batch = batch.to(device)
        preds_scaled = model(batch)

        preds_original = scaler.inverse_transform(preds_scaled.cpu().numpy())

        all_preds.append(preds_original)
        all_ids.append(batch.id.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Ensure ID is integer and sort
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df = submission_df.sort_values("id")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    # load_cached_data=True will try to load from ./working/idea_23/cache/
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=True
    )

    # Initialize Model
    model = RA_CGN(Config).to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # Training Loop
    best_val_rmsle = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_rmsle = evaluate(model, val_loader, criterion, device, scaler)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val RMSLE: {val_rmsle:.6f}"
        )

        # Early Stopping based on RMSLE (Competition Metric)
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch}. Best Val RMSLE: {best_val_rmsle:.6f}"
            )
            break

    # Load best model for submission
    print("Loading best model for submission...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    else:
        print("Warning: Best model checkpoint not found. Using current model.")

    # Generate Submission
    generate_submission(model, test_loader, device, scaler, Config.SUBMISSION_PATH)
