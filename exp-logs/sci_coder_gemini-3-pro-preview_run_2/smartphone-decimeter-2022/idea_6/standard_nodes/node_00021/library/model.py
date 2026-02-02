import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json

from library.config import Config
from library.utils import lla_to_ecef, ecef_to_lla, ecef_to_enu, enu_to_ecef

# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------


class RelativeTrajectoryCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        fc_dim,
        dropout,
        output_dim=2,
    ):
        super(RelativeTrajectoryCNN, self).__init__()

        layers = []
        in_c = input_channels

        # Convolutional Backbone
        for out_c in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_c = out_c

        self.backbone = nn.Sequential(*layers)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(in_c, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, output_dim),
        )

    def forward(self, x):
        # x shape: (Batch, Window_Size, Channels) -> Permute to (Batch, Channels, Window_Size)
        x = x.permute(0, 2, 1)

        features = self.backbone(x)
        pooled = self.gap(features).squeeze(-1)
        output = self.head(pooled)

        return output


# -------------------------------------------------------------------------
# Dataset Definition
# -------------------------------------------------------------------------


class GNSSWindowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# -------------------------------------------------------------------------
# Data Processing Pipeline
# -------------------------------------------------------------------------


def aggregate_gnss_data(gnss_df):
    """
    Aggregates raw GNSS data by epoch (utcTimeMillis).
    """
    # Select columns that exist
    agg_dict = {
        "Svid": "count",
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Filter for available columns
    available_cols = set(gnss_df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in available_cols}

    # Group
    grouped = gnss_df.groupby(["tripId", "utcTimeMillis"]).agg(agg_dict).reset_index()

    # Rename
    rename_map = {
        "Svid": "SatelliteCount",
        "Cn0DbHz": "MeanCn0",
        "RawPseudorangeUncertaintyMeters": "MeanUncertainty",
    }
    grouped.rename(columns=rename_map, inplace=True)

    return grouped


def create_windows(df, window_size):
    """
    Creates sliding windows for the dataset.
    Returns a numpy array of shape (N, window_size, features).
    """
    # Features to window
    # We use Lat/Lon/Alt directly, plus velocities and signal metrics
    feature_cols = [
        "WlsLat",
        "WlsLon",
        "WlsAlt",
        "VelLat",
        "VelLon",
        "VelAlt",
        "MeanCn0",
        "MeanUncertainty",
        "SatelliteCount",
    ]

    # Ensure data is sorted by trip and time
    df = df.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

    # We need to pad per trip to maintain output size = input size
    pad = window_size // 2

    windows_list = []

    for trip_id, group in df.groupby("tripId"):
        data = group[feature_cols].values

        # Pad the beginning and end of the trip sequence
        pad_start = np.tile(data[0], (pad, 1))
        pad_end = np.tile(data[-1], (pad, 1))
        padded_data = np.vstack([pad_start, data, pad_end])

        # Create sliding windows
        stride0, stride1 = padded_data.strides
        num_windows = len(data)

        windows = np.lib.stride_tricks.as_strided(
            padded_data,
            shape=(num_windows, window_size, len(feature_cols)),
            strides=(stride0, stride0, stride1),
        )

        windows_list.append(windows)

    return np.vstack(windows_list)


def process_features(df):
    """
    Generates relative metric features (ENU) and velocities.
    Cite solution_lesson_node_00020: Minimize Coordinate Transformations.
    We use simple scaling of Lat/Lon instead of ECEF->ENU to avoid instability.
    """
    # Ensure WLS columns are filled
    cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df[cols] = df[cols].fillna(0)

    # Convert WLS ECEF to LLA
    wls_lat, wls_lon, wls_alt = ecef_to_lla(
        df["WlsPositionXEcefMeters"].values,
        df["WlsPositionYEcefMeters"].values,
        df["WlsPositionZEcefMeters"].values,
    )

    df["WlsLat"] = wls_lat
    df["WlsLon"] = wls_lon
    df["WlsAlt"] = wls_alt

    # Calculate Velocities (First order difference) in meters approx
    deg_to_m = 111320.0

    # Diffs
    dLat = df.groupby("tripId")["WlsLat"].diff().fillna(0)
    dLon = df.groupby("tripId")["WlsLon"].diff().fillna(0)
    dAlt = df.groupby("tripId")["WlsAlt"].diff().fillna(0)

    # Scale velocities to meters
    df["VelLat"] = dLat * deg_to_m
    df["VelLon"] = dLon * deg_to_m * np.cos(np.radians(df["WlsLat"]))
    df["VelAlt"] = dAlt

    # Fill missing signal metrics
    df["MeanCn0"] = df["MeanCn0"].fillna(0)
    df["MeanUncertainty"] = df["MeanUncertainty"].fillna(100)
    df["SatelliteCount"] = df["SatelliteCount"].fillna(0)

    return df


def re_center_windows(windows):
    """
    Adjusts the position features to be relative to the center of the window
    and scales them to meters.
    """
    # windows shape: (N, W, F)
    # Indices: 0:Lat, 1:Lon, 2:Alt

    center_idx = windows.shape[1] // 2
    center_pos = windows[:, center_idx : center_idx + 1, 0:3]

    # Subtract center from all positions in window
    windows[:, :, 0:3] -= center_pos

    # Scale to meters
    deg_to_m = 111320.0

    # RelLat (North)
    windows[:, :, 0] *= deg_to_m

    # RelLon (East) - use center lat for cosine scaling
    center_lats_rad = np.radians(center_pos[:, :, 0])
    cos_lat = np.cos(center_lats_rad)
    windows[:, :, 1] *= deg_to_m * cos_lat

    # RelAlt is already in meters (diff of meters)

    return windows


def prepare_data(metadata_path, mode="train", load_cached_data=True, debug_size=None):
    """
    Main data preparation function.
    """
    # Define cache paths based on mode
    if mode == "train":
        x_cache = Config.TRAIN_X_CACHE
        y_cache = Config.TRAIN_Y_CACHE
        meta_cache = Config.TRAIN_META_CACHE
    elif mode == "val":
        x_cache = Config.VAL_X_CACHE
        y_cache = Config.VAL_Y_CACHE
        meta_cache = Config.VAL_META_CACHE
    else:
        x_cache = Config.TEST_X_CACHE
        y_cache = None
        meta_cache = Config.TEST_META_CACHE

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(x_cache) and os.path.exists(meta_cache):
        print(f"Loading cached data for {mode}...")
        X = np.load(x_cache)
        df_meta = pd.read_parquet(meta_cache)
        y = np.load(y_cache) if y_cache and os.path.exists(y_cache) else None

        if debug_size:
            return (
                X[:debug_size],
                y[:debug_size] if y is not None else None,
                df_meta.iloc[:debug_size],
            )
        return X, y, df_meta

    # 2. Process from Scratch
    print(f"Processing data for {mode} from scratch...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)
    if debug_size:
        df_meta = df_meta.iloc[:debug_size]

    unique_trips = df_meta["tripId"].unique()

    # Load and aggregate GNSS data
    gnss_dfs = []
    for trip_id in unique_trips:
        # Find path from metadata (first row of trip)
        row = df_meta[df_meta["tripId"] == trip_id].iloc[0]
        gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

        if os.path.exists(gnss_path):
            g_df = pd.read_csv(gnss_path)
            g_df["tripId"] = trip_id
            gnss_dfs.append(g_df)

    if not gnss_dfs:
        raise ValueError("No GNSS data found.")

    full_gnss = pd.concat(gnss_dfs, ignore_index=True)
    agg_gnss = aggregate_gnss_data(full_gnss)

    # Merge with metadata (Ground Truth or Test timestamps)
    # We use left join on metadata to ensure we have rows for all required predictions
    merged = pd.merge(
        df_meta,
        agg_gnss,
        left_on=["tripId", "UnixTimeMillis"],
        right_on=["tripId", "utcTimeMillis"],
        how="left",
    )

    # Forward fill missing WLS/Signal data within trips (for gaps)
    # Sort first
    merged = merged.sort_values(["tripId", "UnixTimeMillis"])
    cols_to_ffill = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "MeanCn0",
        "MeanUncertainty",
        "SatelliteCount",
    ]
    merged[cols_to_ffill] = merged.groupby("tripId")[cols_to_ffill].ffill().bfill()

    # Feature Engineering
    merged = process_features(merged)

    # Create Windows
    X = create_windows(merged, Config.WINDOW_SIZE)

    # Re-center windows (Make position relative to center)
    X = re_center_windows(X)

    # Targets (for train/val)
    y = None
    if mode in ["train", "val"]:
        # Calculate Ground Truth ENU relative to WLS
        # We use the WLS position of the current epoch as reference
        # Target = GT - WLS (in meters)

        # GT LLA
        gt_lat = merged["LatitudeDegrees"].values
        gt_lon = merged["LongitudeDegrees"].values
        gt_alt = merged["AltitudeMeters"].values

        # WLS LLA
        wls_lat = merged["WlsLat"].values
        wls_lon = merged["WlsLon"].values
        wls_alt = merged["WlsAlt"].values

        # Convert GT LLA to ECEF
        gt_x, gt_y, gt_z = lla_to_ecef(gt_lat, gt_lon, gt_alt)

        # Convert GT ECEF to ENU relative to WLS LLA
        t_e, t_n, t_u = ecef_to_enu(gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt)

        # Target: Delta East, Delta North
        y = np.stack([t_e, t_n], axis=1)

        # Filter outliers (optional, but good for stability)
        # mask = (np.abs(y) < 1000).all(axis=1) # Filter > 1km errors?
        # For now, keep all, robust loss handles outliers.

    # Scaling
    # We fit scaler on Train, apply to Val/Test
    # Reshape X to (N*W, F) for scaling
    N, W, F = X.shape
    X_flat = X.reshape(-1, F)

    if mode == "train":
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
        # Save scaler params
        scaler_params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        with open(Config.SCALER_PATH, "w") as f:
            json.dump(scaler_params, f)
    else:
        # Load scaler
        if os.path.exists(Config.SCALER_PATH):
            with open(Config.SCALER_PATH, "r") as f:
                params = json.load(f)
            scaler = StandardScaler()
            scaler.mean_ = np.array(params["mean"])
            scaler.scale_ = np.array(params["scale"])
            X_flat = scaler.transform(X_flat)
        else:
            print("Warning: Scaler not found, using local scaling (suboptimal).")
            scaler = StandardScaler()
            X_flat = scaler.fit_transform(X_flat)

    X = X_flat.reshape(N, W, F)

    # Save Cache
    np.save(x_cache, X)
    merged.to_parquet(meta_cache, index=False)
    if y is not None:
        np.save(y_cache, y)

    return X, y, merged


# -------------------------------------------------------------------------
# Training & Inference
# -------------------------------------------------------------------------


def train_model(debug_size=None, epochs=Config.NUM_EPOCHS):
    # Load Data
    X_train, y_train, _ = prepare_data(
        Config.TRAIN_METADATA_PATH, "train", debug_size=debug_size
    )
    X_val, y_val, _ = prepare_data(
        Config.VAL_METADATA_PATH, "val", debug_size=debug_size
    )

    # Dataset & Loader
    train_ds = GNSSWindowDataset(X_train, y_train)
    val_ds = GNSSWindowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model
    model = RelativeTrajectoryCNN(
        input_channels=Config.NUM_INPUT_FEATURES,
        hidden_channels=Config.CNN_HIDDEN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        fc_dim=Config.FC_HIDDEN_DIM,
        dropout=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_ds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation MAE: {best_val_loss:.6f}")


def predict_and_submit(debug_size=None):
    # Load Test Data
    X_test, _, df_test_meta = prepare_data(
        Config.TEST_METADATA_PATH, "test", debug_size=debug_size
    )

    test_ds = GNSSWindowDataset(X_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = RelativeTrajectoryCNN(
        input_channels=Config.NUM_INPUT_FEATURES,
        hidden_channels=Config.CNN_HIDDEN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        fc_dim=Config.FC_HIDDEN_DIM,
        dropout=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train first.")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.vstack(predictions)  # (N, 2) -> Delta East, Delta North

    # Reconstruct Absolute Coordinates
    # Pred Lat/Lon = WLS Lat/Lon + Convert(Delta East, Delta North)

    wls_lat = df_test_meta["WlsLat"].values
    wls_lon = df_test_meta["WlsLon"].values
    wls_alt = df_test_meta["WlsAlt"].values  # Use WLS Alt as reference

    # We need to convert the predicted ENU offsets back to ECEF, then to LLA
    # But wait, we just need to add the offset.
    # Target was: ENU(GT) relative to WLS.
    # Prediction is: ENU offset.
    # So Reconstructed Position = ENU_to_LLA(Prediction, Ref=WLS)

    pred_e = predictions[:, 0]
    pred_n = predictions[:, 1]
    pred_u = np.zeros_like(
        pred_e
    )  # We didn't predict Up, assume 0 offset (or could predict it)

    # Vectorized reconstruction
    # Note: enu_to_lla_relative is defined in utils but uses scalar logic in loop usually
    # We use the vectorized version provided in utils.py

    # Convert ENU offsets to ECEF
    # We need to call enu_to_ecef.
    # Since utils functions are imported, we use them.
    # However, utils.enu_to_ecef expects arrays.

    pred_x, pred_y, pred_z = enu_to_ecef(
        pred_e, pred_n, pred_u, wls_lat, wls_lon, wls_alt
    )

    pred_lat, pred_lon, _ = ecef_to_lla(pred_x, pred_y, pred_z)

    # Create Submission
    submission = pd.DataFrame(
        {
            "tripId": df_test_meta["tripId"],
            "UnixTimeMillis": df_test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
