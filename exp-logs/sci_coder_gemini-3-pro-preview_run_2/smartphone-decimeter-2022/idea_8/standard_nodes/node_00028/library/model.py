import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.dataset import get_dataloaders
from library.utils import ecef_to_lla, meters_to_degrees_diff

# -------------------------------------------------------------------------
# 1. Model Architecture
# -------------------------------------------------------------------------


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism to aggregate sequence of hidden states.
    Computes a weighted sum of the hidden states based on learned importance.
    """

    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)

        # Compute attention scores
        # scores shape: (batch_size, seq_len, 1)
        scores = self.query(x)

        # Compute weights
        weights = F.softmax(scores, dim=1)

        # Weighted sum
        # context shape: (batch_size, hidden_size)
        context = torch.sum(x * weights, dim=1)

        return context


class BiGRUModel(nn.Module):
    """
    Bidirectional GRU with Temporal Attention for relative state regression.
    """

    def __init__(self):
        super(BiGRUModel, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS
        self.dropout_prob = Config.DROPOUT
        self.output_dim = Config.OUTPUT_DIM
        self.bidirectional = Config.BIDIRECTIONAL

        # Bidirectional GRU Encoder
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        # Calculate effective hidden size after direction concatenation
        self.gru_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Temporal Attention Layer
        self.attention = TemporalAttention(self.gru_output_size)

        # Regression Head (MLP)
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim),
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, input_dim)

        # Pass through GRU
        # gru_out shape: (batch_size, window_size, gru_output_size)
        gru_out, _ = self.gru(x)

        # Apply Attention
        # context shape: (batch_size, gru_output_size)
        context = self.attention(gru_out)

        # Predict residuals
        # out shape: (batch_size, output_dim)
        out = self.fc(context)

        return out


# -------------------------------------------------------------------------
# 2. Training Logic
# -------------------------------------------------------------------------


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate MAE for metrics
    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    mae = mean_absolute_error(all_targets, all_preds)

    return epoch_loss, mae


def train_model(train_loader, val_loader):
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    model = BiGRUModel().to(device)

    # Using L1 Loss (MAE) for robustness against outliers
    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"Time: {duration:.1f}s"
        )

        # Early Stopping and Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> Model saved! Best Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("  -> Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    return model


# -------------------------------------------------------------------------
# 3. Inference and Submission Logic
# -------------------------------------------------------------------------


def get_wls_baselines(test_meta_df):
    """
    Retrieves WLS baseline positions for the test set.
    Since the preprocessor strips absolute coordinates, we need to fetch
    the WLS baseline (Lat, Lon) for each test timestamp to add the predicted residuals.
    """
    print("Retrieving WLS baselines for test set...")
    baselines = []

    # Group by trip to minimize file I/O
    for trip_id, group in test_meta_df.groupby("tripId"):
        # Get path from the first row of the group
        gnss_rel_path = group.iloc[0]["gnss_path"]
        gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            # Fallback (should not happen in valid test set)
            print(f"Warning: GNSS file not found for {trip_id}")
            group["wls_lat"] = 0.0
            group["wls_lon"] = 0.0
            baselines.append(group[["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]])
            continue

        # Load GNSS data
        try:
            df_gnss = pd.read_csv(gnss_path)

            # Aggregate to epoch level (WLS positions are repeated per signal, take first)
            df_epoch = (
                df_gnss.groupby("utcTimeMillis")
                .agg(
                    {
                        "WlsPositionXEcefMeters": "first",
                        "WlsPositionYEcefMeters": "first",
                        "WlsPositionZEcefMeters": "first",
                    }
                )
                .reset_index()
            )

            # Convert ECEF to LLA
            lats, lons, _ = ecef_to_lla(
                df_epoch["WlsPositionXEcefMeters"].values,
                df_epoch["WlsPositionYEcefMeters"].values,
                df_epoch["WlsPositionZEcefMeters"].values,
            )
            df_epoch["wls_lat"] = lats
            df_epoch["wls_lon"] = lons

            # Interpolate to match target timestamps exactly
            # Create a full index covering the range
            df_epoch = df_epoch.set_index("utcTimeMillis")

            # Reindex to the specific target timestamps requested in test_meta
            target_ts = group["UnixTimeMillis"].values

            # We combine existing indices with target indices to interpolate
            combined_index = np.unique(
                np.concatenate([df_epoch.index.values, target_ts])
            )
            combined_index.sort()

            df_interp = (
                df_epoch.reindex(combined_index)
                .interpolate(method="index")
                .loc[target_ts]
            )

            # Assign back
            group_res = group.copy()
            group_res["wls_lat"] = df_interp["wls_lat"].values
            group_res["wls_lon"] = df_interp["wls_lon"].values

            baselines.append(
                group_res[["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]]
            )

        except Exception as e:
            print(f"Error processing {trip_id}: {e}")
            # Fill with 0 or NaNs
            group["wls_lat"] = 0.0
            group["wls_lon"] = 0.0
            baselines.append(group[["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]])

    # Concatenate all baselines
    df_baselines = pd.concat(baselines, ignore_index=True)

    # Ensure order matches test_meta
    df_baselines = pd.merge(
        test_meta_df[["tripId", "UnixTimeMillis"]],
        df_baselines,
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    return df_baselines


def generate_submission(model, test_loader, test_meta):
    device = torch.device(Config.DEVICE)
    model.eval()

    print("Generating predictions...")
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    # Concatenate all predictions (N_samples, 2) -> [d_lat_m, d_lon_m]
    pred_residuals = np.concatenate(predictions, axis=0)

    # Get WLS Baselines
    df_baselines = get_wls_baselines(test_meta)

    # Reconstruct absolute coordinates
    # pred_lat = wls_lat + meters_to_deg(d_lat_m)
    # pred_lon = wls_lon + meters_to_deg(d_lon_m)

    wls_lats = df_baselines["wls_lat"].values
    wls_lons = df_baselines["wls_lon"].values

    d_lat_m = pred_residuals[:, 0]
    d_lon_m = pred_residuals[:, 1]

    d_lat_deg, d_lon_deg = meters_to_degrees_diff(d_lat_m, d_lon_m, wls_lats)

    final_lats = wls_lats + d_lat_deg
    final_lons = wls_lons + d_lon_deg

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": final_lats,
            "LongitudeDegrees": final_lons,
        }
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


def run_pipeline():
    # 1. Get Data
    train_loader, val_loader, test_loader, test_meta = get_dataloaders(
        load_cached_data=True
    )

    # 2. Train Model
    model = train_model(train_loader, val_loader)

    # 3. Generate Submission
    generate_submission(model, test_loader, test_meta)
