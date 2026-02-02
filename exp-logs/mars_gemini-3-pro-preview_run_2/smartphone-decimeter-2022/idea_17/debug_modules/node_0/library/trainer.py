import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_PATH,
    SEED,
    SUBMISSION_DIR,
)
from library.model import SCRCNN
from library.dataset import get_dataloaders
from library.utils import get_local_scale_factors


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


def train_model(load_cached_data=True):
    """
    Trains the SCRCNN model.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.

    Returns:
        model (nn.Module): The trained model (best state).
    """
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get DataLoaders
    # Unpack the tuple returned by get_dataloaders
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Initialize Model
    model = SCRCNN().to(device)

    # Loss and Optimizer
    # L1 Loss (MAE) is robust to outliers
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for x_kin, x_sky, y in train_loader:
            x_kin = x_kin.to(device)
            x_sky = x_sky.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x_kin, x_sky)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_kin.size(0)
            train_samples += x_kin.size(0)

        train_loss /= train_samples

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for x_kin, x_sky, y in val_loader:
                x_kin = x_kin.to(device)
                x_sky = x_sky.to(device)
                y = y.to(device)

                outputs = model(x_kin, x_sky)
                loss = criterion(outputs, y)

                val_loss += loss.item() * x_kin.size(0)
                val_samples += x_kin.size(0)

        val_loss /= val_samples

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved to {MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model weights before returning
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    return model


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the trained model.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating submission...")

    # Load Test Data
    # We ignore train/val loaders here
    _, _, test_loader, test_meta = get_dataloaders(load_cached_data=load_cached_data)

    # Initialize Model
    model = SCRCNN().to(device)

    # Load Weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(
            f"Warning: No trained model found at {MODEL_PATH}. Predictions will be random."
        )

    model.eval()

    preds_list = []

    # Inference Loop
    with torch.no_grad():
        for x_kin, x_sky in test_loader:
            x_kin = x_kin.to(device)
            x_sky = x_sky.to(device)

            outputs = model(x_kin, x_sky)
            preds_list.append(outputs.cpu().numpy())

    # Concatenate predictions (N, 2) -> [dLat_m, dLon_m]
    preds = np.concatenate(preds_list, axis=0)

    # Reconstruction: Convert metric residuals back to degrees
    # Get baseline WLS positions from metadata
    wls_lats = test_meta["wls_lat"].values
    wls_lons = test_meta["wls_lon"].values

    # Calculate local scale factors (meters per degree)
    lat_scales, lon_scales = get_local_scale_factors(wls_lats)

    # Convert meters to degrees
    # dDeg = dMeters / Scale
    d_lat_deg = preds[:, 0] / lat_scales
    d_lon_deg = preds[:, 1] / lon_scales

    # Add residuals to baseline
    pred_lats = wls_lats + d_lat_deg
    pred_lons = wls_lons + d_lon_deg

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )

    # Save Submission
    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
