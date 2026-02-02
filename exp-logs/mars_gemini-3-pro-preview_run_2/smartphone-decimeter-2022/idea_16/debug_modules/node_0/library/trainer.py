import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    NUM_WORKERS,
    CACHE_DIR,
    SUBMISSION_DIR,
    SAMPLE_SUBMISSION_PATH,
)
from library.model import RelativeWindowedMLP
from library.data_loader import load_dataset
from library.utils import meters_to_wgs84_relative


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for batch in dataloader:
        traj = batch["traj_feat"].to(device)
        sky = batch["sky_feat"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(traj, sky)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * traj.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for batch in dataloader:
            traj = batch["traj_feat"].to(device)
            sky = batch["sky_feat"].to(device)
            targets = batch["target"].to(device)

            outputs = model(traj, sky)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * traj.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_training(max_epochs=EPOCHS, max_samples=None, load_cached=True):
    """
    Main training loop with early stopping and model checkpointing.
    """
    print(f"Loading datasets (max_samples={max_samples})...")
    train_dataset, scaler = load_dataset(
        mode="train", max_samples=max_samples, load_cached_data=load_cached
    )
    val_dataset, _ = load_dataset(
        mode="val", scaler=scaler, max_samples=max_samples, load_cached_data=load_cached
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print(f"Initializing model on {DEVICE}...")
    model = RelativeWindowedMLP().to(DEVICE)

    # L1 Loss (MAE) is robust to outliers common in GNSS data
    criterion = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{max_epochs} - "
            f"Train MAE: {train_loss:.10f} - "
            f"Val MAE: {val_loss:.10f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with Val MAE: {best_val_loss:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    return scaler


def run_inference(scaler=None, load_cached=True):
    """
    Generates predictions for the test set and creates a submission file.
    """
    print("Loading test dataset...")
    # If scaler is None, load_dataset will attempt to load it from disk
    test_dataset, _ = load_dataset(
        mode="test", scaler=scaler, load_cached_data=load_cached
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print("Loading best model...")
    model = RelativeWindowedMLP().to(DEVICE)
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    print("Generating predictions...")
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            traj = batch["traj_feat"].to(DEVICE)
            sky = batch["sky_feat"].to(DEVICE)

            # Output is [delta_x, delta_y] in meters
            outputs = model(traj, sky)
            preds.append(outputs.cpu().numpy())

    pred_residuals = np.concatenate(preds, axis=0)

    # Reconstruction: Convert metric residuals back to WGS84
    test_meta = test_dataset.meta  # [trip_id, timestamp, wls_lat, wls_lon]
    pred_lats = []
    pred_lons = []

    print("Reconstructing WGS84 coordinates...")
    for i in range(len(test_meta)):
        wls_lat = test_meta[i, 2]
        wls_lon = test_meta[i, 3]

        dx = pred_residuals[i, 0]  # Easting offset
        dy = pred_residuals[i, 1]  # Northing offset

        lat, lon = meters_to_wgs84_relative(wls_lat, wls_lon, dx, dy)
        pred_lats.append(lat)
        pred_lons.append(lon)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta[:, 0],
            "UnixTimeMillis": test_meta[:, 1],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )

    # Ensure correct types for merging
    submission_df["UnixTimeMillis"] = submission_df["UnixTimeMillis"].astype(np.int64)

    # Merge with sample submission to ensure correct order and completeness
    print("Creating submission file...")
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
        submission_df, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Fill any missing predictions with sample submission values (fallback)
    final_sub["LatitudeDegrees"] = final_sub["LatitudeDegrees"].fillna(
        sample_sub["LatitudeDegrees"]
    )
    final_sub["LongitudeDegrees"] = final_sub["LongitudeDegrees"].fillna(
        sample_sub["LongitudeDegrees"]
    )

    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
