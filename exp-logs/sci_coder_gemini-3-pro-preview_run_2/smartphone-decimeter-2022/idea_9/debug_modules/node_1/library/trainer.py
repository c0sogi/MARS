import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    RANDOM_STATE,
    DEG_TO_M_LAT,
    DEG_TO_M_LON,
)
from library.data_loader import get_dataset
from library.preprocessing import GNSSScaler, GNSSSequenceDataset
from library.model import GeometryConditionedCNN
from library.utils import meters_to_degrees


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(load_cached_data: bool = True):
    """
    Train the Geometry-Conditioned Residual 1D-CNN model.

    Args:
        load_cached_data: Whether to load pre-processed data/scaler from cache.

    Returns:
        model: The trained PyTorch model (loaded with best weights).
        scaler: The fitted GNSSScaler.
    """
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # get_dataset handles parquet caching internally based on load_cached_data flag
    train_df = get_dataset("train", load_cached_data=load_cached_data)
    val_df = get_dataset("val", load_cached_data=load_cached_data)

    # 2. Prepare Scaler
    scaler = GNSSScaler()
    scaler_path = os.path.join(WORKING_DIR, "scaler.json")

    if load_cached_data and os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        scaler.load(scaler_path)
    else:
        print("Fitting scaler on training data...")
        scaler.fit(train_df)
        print(f"Saving scaler to {scaler_path}")
        scaler.save(scaler_path)

    # 3. Create Datasets and Loaders
    train_dataset = GNSSSequenceDataset(train_df, scaler, is_test=False)
    val_dataset = GNSSSequenceDataset(val_df, scaler, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = GeometryConditionedCNN().to(device)

    # 5. Setup Training Components
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for traj, ctx, target in train_loader:
            traj = traj.to(device)
            ctx = ctx.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(traj, ctx)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * traj.size(0)

        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for traj, ctx, target in val_loader:
                traj = traj.to(device)
                ctx = ctx.to(device)
                target = target.to(device)

                output = model(traj, ctx)
                loss = criterion(output, target)

                val_loss += loss.item() * traj.size(0)

        val_loss /= len(val_dataset)

        # Log metrics with full precision
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model weights
    if os.path.exists(best_model_path):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(best_model_path))

    return model, scaler


def predict_and_submit(model, scaler, load_cached_data: bool = True):
    """
    Generate predictions for the test set and save the submission file.

    Args:
        model: Trained PyTorch model.
        scaler: Fitted GNSSScaler.
        load_cached_data: Whether to load pre-processed test data from cache.
    """
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using device: {device}")

    # 1. Load Test Data
    test_df = get_dataset("test", load_cached_data=load_cached_data)

    # 2. Create Dataset and Loader
    test_dataset = GNSSSequenceDataset(test_df, scaler, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Inference
    model.eval()
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for traj, ctx in test_loader:
            traj = traj.to(device)
            ctx = ctx.to(device)

            output = model(traj, ctx)
            all_preds.append(output.cpu().numpy())

    # Concatenate all predictions [N, 2] -> (res_lat_m, res_lon_m)
    pred_residuals_m = np.concatenate(all_preds, axis=0)

    # 4. Reconstruction
    # We need to add the predicted residuals (in meters) back to the WLS baseline (in degrees).
    # The test_df is guaranteed to be in the same order as the dataset iteration
    # because GNSSSequenceDataset iterates tripIds in the order they appear in test_df.

    wls_lat = test_df["wls_lat"].values
    wls_lon = test_df["wls_lon"].values

    pred_res_lat_m = pred_residuals_m[:, 0]
    pred_res_lon_m = pred_residuals_m[:, 1]

    # Convert metric residuals to degrees
    # Note: meters_to_degrees handles the cosine scaling for longitude
    pred_lat_deg, pred_lon_deg = meters_to_degrees(
        pred_res_lat_m, pred_res_lon_m, wls_lat
    )

    final_lat = wls_lat + pred_lat_deg
    final_lon = wls_lon + pred_lon_deg

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_df["tripId"],
            "UnixTimeMillis": test_df["UnixTimeMillis"],
            "LatitudeDegrees": final_lat,
            "LongitudeDegrees": final_lon,
        }
    )

    # 6. Save
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")
