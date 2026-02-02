import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, meters_to_latlon, haversine_distance
from library.data_loader import get_dataloaders
from library.model import SkyContextualizedCNN


def train_model(load_cached_data=True):
    """
    Trains the Sky-Contextualized CNN model.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False, processes data from scratch.
    """
    # Set random seed
    seed_everything(Config.RANDOM_STATE)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    # val_meta and test_meta are pandas DataFrames aligned with the loaders
    train_loader, val_loader, test_loader, val_meta, test_meta = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Model
    model = SkyContextualizedCNN().to(device)

    # Loss and Optimizer
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training Loop
    best_val_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for traj, sky, target in train_loader:
            traj = traj.to(device)
            sky = sky.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(traj, sky)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * traj.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for traj, sky, target in val_loader:
                traj = traj.to(device)
                sky = sky.to(device)

                output = model(traj, sky)

                val_preds_list.append(output.cpu().numpy())
                val_targets_list.append(target.numpy())

        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)

        # Calculate Validation Metrics
        # We need to reconstruct Lat/Lon to calculate Haversine distance
        # val_meta contains 'wls_lat', 'wls_lon' aligned with the predictions
        wls_lat = val_meta["wls_lat"].values
        wls_lon = val_meta["wls_lon"].values

        # Reconstruct Ground Truth Lat/Lon from targets (d_east, d_north)
        gt_lat, gt_lon = meters_to_latlon(
            wls_lat, wls_lon, val_targets[:, 0], val_targets[:, 1]
        )

        # Reconstruct Predicted Lat/Lon from preds (d_east, d_north)
        pred_lat, pred_lon = meters_to_latlon(
            wls_lat, wls_lon, val_preds[:, 0], val_preds[:, 1]
        )

        # Calculate Haversine distances
        distances = haversine_distance(gt_lat, gt_lon, pred_lat, pred_lon)

        mean_dist_error = np.mean(distances)
        p50 = np.percentile(distances, 50)
        p95 = np.percentile(distances, 95)
        val_score = (p50 + p95) / 2.0

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Mean Dist: {mean_dist_error:.6f} | "
            f"Val Score (50+95)/2: {val_score:.6f}"
        )

        # Learning Rate Scheduler
        scheduler.step(val_score)

        # Early Stopping and Checkpointing
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_val_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # --- Prediction Phase ---
    print("Generating submission for test set...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()

    test_preds_list = []

    with torch.no_grad():
        for traj, sky in test_loader:
            traj = traj.to(device)
            sky = sky.to(device)

            output = model(traj, sky)
            test_preds_list.append(output.cpu().numpy())

    test_preds = np.concatenate(test_preds_list, axis=0)

    # Reconstruct Test Coordinates
    t_wls_lat = test_meta["wls_lat"].values
    t_wls_lon = test_meta["wls_lon"].values

    pred_test_lat, pred_test_lon = meters_to_latlon(
        t_wls_lat, t_wls_lon, test_preds[:, 0], test_preds[:, 1]
    )

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["utcTimeMillis"],
            "LatitudeDegrees": pred_test_lat,
            "LongitudeDegrees": pred_test_lon,
        }
    )

    # Save Submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
