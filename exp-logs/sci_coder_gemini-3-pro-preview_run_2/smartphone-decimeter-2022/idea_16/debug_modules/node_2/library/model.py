import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    INPUT_DIM_TRAJ,
    INPUT_DIM_SKY,
    OUTPUT_DIM,
    TRAJ_HIDDEN_DIMS,
    SKY_HIDDEN_DIMS,
    FUSION_HIDDEN_DIMS,
    DROPOUT,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    NUM_WORKERS,
    LAT_METERS_PER_DEGREE,
    SUBMISSION_DIR,
    SAMPLE_SUBMISSION_PATH,
    CACHE_DIR,
)
from library.data_loader import load_dataset
from library.utils import meters_to_wgs84_relative


class RelativeWindowedMLP(nn.Module):
    def __init__(self):
        super(RelativeWindowedMLP, self).__init__()

        # --- Trajectory Encoder ---
        traj_layers = []
        in_dim = INPUT_DIM_TRAJ
        for hidden_dim in TRAJ_HIDDEN_DIMS:
            traj_layers.append(nn.Linear(in_dim, hidden_dim))
            traj_layers.append(nn.BatchNorm1d(hidden_dim))
            traj_layers.append(nn.ReLU())
            traj_layers.append(nn.Dropout(DROPOUT))
            in_dim = hidden_dim
        self.traj_net = nn.Sequential(*traj_layers)
        self.traj_out_dim = TRAJ_HIDDEN_DIMS[-1]

        # --- Sky Context Encoder ---
        sky_layers = []
        in_dim = INPUT_DIM_SKY
        for hidden_dim in SKY_HIDDEN_DIMS:
            sky_layers.append(nn.Linear(in_dim, hidden_dim))
            sky_layers.append(nn.BatchNorm1d(hidden_dim))
            sky_layers.append(nn.ReLU())
            sky_layers.append(nn.Dropout(DROPOUT))
            in_dim = hidden_dim
        self.sky_net = nn.Sequential(*sky_layers)
        self.sky_out_dim = SKY_HIDDEN_DIMS[-1]

        # --- Fusion & Prediction ---
        fusion_layers = []
        in_dim = self.traj_out_dim + self.sky_out_dim
        for hidden_dim in FUSION_HIDDEN_DIMS:
            fusion_layers.append(nn.Linear(in_dim, hidden_dim))
            fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(DROPOUT))
            in_dim = hidden_dim

        # Output layer (no activation, regression)
        fusion_layers.append(nn.Linear(in_dim, OUTPUT_DIM))
        self.fusion_net = nn.Sequential(*fusion_layers)

    def forward(self, traj_feat, sky_feat):
        # traj_feat: [batch, input_dim_traj]
        # sky_feat: [batch, input_dim_sky]

        traj_emb = self.traj_net(traj_feat)
        sky_emb = self.sky_net(sky_feat)

        # Concatenate
        combined = torch.cat([traj_emb, sky_emb], dim=1)

        # Predict residuals (meters)
        output = self.fusion_net(combined)
        return output


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

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

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            traj = batch["traj_feat"].to(device)
            sky = batch["sky_feat"].to(device)
            targets = batch["target"].to(device)

            outputs = model(traj, sky)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * traj.size(0)

    return running_loss / len(dataloader.dataset)


def predict(model, dataloader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            traj = batch["traj_feat"].to(device)
            sky = batch["sky_feat"].to(device)

            outputs = model(traj, sky)
            preds.append(outputs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_pipeline():
    print("Starting Relative Windowed MLP Pipeline...")

    # 1. Load Data
    print("Loading Train Data...")
    train_dataset, scaler = load_dataset(mode="train", load_cached_data=True)
    print("Loading Validation Data...")
    val_dataset, _ = load_dataset(mode="val", scaler=scaler, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 2. Initialize Model
    model = RelativeWindowedMLP().to(DEVICE)
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train MAE: {train_loss:.6f} - Val MAE: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 4. Inference
    print("Loading Test Data...")
    test_dataset, _ = load_dataset(mode="test", scaler=scaler, load_cached_data=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    print("Generating predictions...")

    # Predict residuals (meters)
    pred_residuals = predict(model, test_loader, DEVICE)

    # 5. Reconstruction
    # Get metadata to retrieve WLS centers
    # Dataset meta is numpy array: [trip_id, timestamp, wls_lat, wls_lon]
    test_meta = test_dataset.meta

    pred_lats = []
    pred_lons = []

    for i in range(len(test_meta)):
        wls_lat = test_meta[i, 2]
        wls_lon = test_meta[i, 3]

        dx = pred_residuals[i, 0]  # East
        dy = pred_residuals[i, 1]  # North

        # Convert metric offset back to degrees
        # Note: meters_to_wgs84_relative(lat_base, lon_base, delta_x, delta_y)
        lat, lon = meters_to_wgs84_relative(wls_lat, wls_lon, dx, dy)

        pred_lats.append(lat)
        pred_lons.append(lon)

    # 6. Create Submission
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta[:, 0],
            "UnixTimeMillis": test_meta[:, 1],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )

    # Ensure sample submission format
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    # Merge to ensure order and missing rows (though test_metadata should cover it)
    final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
        submission_df, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Fill missing (if any) with sample submission values (which are usually WLS or baseline)
    final_sub["LatitudeDegrees"] = final_sub["LatitudeDegrees"].fillna(
        sample_sub["LatitudeDegrees"]
    )
    final_sub["LongitudeDegrees"] = final_sub["LongitudeDegrees"].fillna(
        sample_sub["LongitudeDegrees"]
    )

    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
