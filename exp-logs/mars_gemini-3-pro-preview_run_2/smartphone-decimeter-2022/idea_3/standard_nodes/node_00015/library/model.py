import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.utils import WGS84_to_ECEF, ECEF_to_ENU, ECEF_to_WGS84, haversine_distance

# -------------------------------------------------------------------
# 1. Model Architecture
# -------------------------------------------------------------------


class WindowedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(WindowedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------
# 2. Data Processing & Dataset
# -------------------------------------------------------------------


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


def get_derived_features(df):
    """
    Compute derived features like deltas.
    Assumes df is sorted by time within a trip.
    """
    # First derivatives (Velocity proxy)
    df["dLat"] = df["WlsLat"].diff().fillna(0)
    df["dLon"] = df["WlsLon"].diff().fillna(0)
    df["dAlt"] = df["WlsAlt"].diff().fillna(0)
    return df


def create_windows(data, window_size):
    """
    Create sliding windows from 2D array.
    data: (N_samples, N_features)
    Returns: (N_samples, window_size, N_features)
    """
    # Simple sliding window using stride_tricks or loop
    # For simplicity and handling edge cases (padding), we iterate
    # We pad the beginning and end to maintain output size = input size
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1

    # Pad with edge values
    data_padded = np.pad(data, ((pad_left, pad_right), (0, 0)), mode="edge")

    # Create windows
    # Shape: (N, W, F)
    # Using stride tricks for efficiency
    sub_shape = (len(data), window_size, data.shape[1])
    view = np.lib.stride_tricks.as_strided(
        data_padded,
        shape=sub_shape,
        strides=(data.strides[0], data.strides[0], data.strides[1]),
    )
    return np.copy(view)  # Return copy to avoid memory issues with strides


def process_data(mode, window_size=11, load_cached_data=True):
    """
    Process raw data into windowed tensors for the model.
    mode: 'train', 'val', or 'test'
    """
    cache_dir = "./working/idea_3/"
    os.makedirs(cache_dir, exist_ok=True)

    X_path = os.path.join(cache_dir, f"{mode}_X.npy")
    y_path = os.path.join(cache_dir, f"{mode}_y.npy")
    meta_path = os.path.join(cache_dir, f"{mode}_meta.parquet")
    scaler_path = os.path.join(cache_dir, "scaler_stats.json")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(X_path) and os.path.exists(meta_path):
        if mode == "test" or os.path.exists(y_path):
            print(f"Loading cached data for {mode}...")
            X = np.load(X_path)
            df_meta = pd.read_parquet(meta_path)
            y = np.load(y_path) if mode != "test" else None
            return X, y, df_meta

    # 2. Process from Scratch
    print(f"Processing data for {mode} from scratch...")

    # Load Metadata
    if mode == "train":
        meta_file = "./metadata/train_metadata.csv"
    elif mode == "val":
        meta_file = "./metadata/validation_metadata.csv"
    else:
        meta_file = "./metadata/test_metadata.csv"

    df_meta_idx = pd.read_csv(meta_file)

    # Define input directory
    input_dir = "./input"

    # Features to aggregate from GNSS
    gnss_agg = {
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "Svid": "count",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    all_X = []
    all_y = []
    all_meta = []

    unique_trips = df_meta_idx["tripId"].unique()

    for trip_id in unique_trips:
        trip_info = df_meta_idx[df_meta_idx["tripId"] == trip_id]

        # Load GNSS
        gnss_rel_path = trip_info.iloc[0]["gnss_path"]
        gnss_path = os.path.join(input_dir, gnss_rel_path)

        if not os.path.exists(gnss_path):
            continue

        df_gnss = pd.read_csv(gnss_path)

        # Aggregate by epoch
        df_epoch = df_gnss.groupby("utcTimeMillis").agg(gnss_agg).reset_index()
        df_epoch.rename(
            columns={
                "Svid": "SatCount",
                "Cn0DbHz": "MeanCn0",
                "RawPseudorangeUncertaintyMeters": "MeanUnc",
            },
            inplace=True,
        )

        # Convert WLS ECEF to Lat/Lon/Alt
        wls_lat, wls_lon, wls_alt = ECEF_to_WGS84(
            df_epoch["WlsPositionXEcefMeters"].values,
            df_epoch["WlsPositionYEcefMeters"].values,
            df_epoch["WlsPositionZEcefMeters"].values,
        )
        df_epoch["WlsLat"] = wls_lat
        df_epoch["WlsLon"] = wls_lon
        df_epoch["WlsAlt"] = wls_alt

        # Merge with Metadata (Ground Truth or Submission Sample)
        # Use merge_asof or exact merge. Metadata is usually exact seconds.
        # Ensure sorting
        df_epoch = df_epoch.sort_values("utcTimeMillis")
        trip_info = trip_info.sort_values("UnixTimeMillis")

        # Exact merge preferred as per dataset description
        df_merged = pd.merge(
            trip_info,
            df_epoch,
            left_on="UnixTimeMillis",
            right_on="utcTimeMillis",
            how="inner",
        )

        if df_merged.empty:
            continue

        # Compute Derived Features (Deltas)
        df_merged = get_derived_features(df_merged)

        # Prepare Features Matrix
        # Dynamic: [dLat, dLon, dAlt, MeanCn0, MeanUnc, SatCount]
        # Static: [WlsLat, WlsLon]
        feature_cols = [
            "dLat",
            "dLon",
            "dAlt",
            "MeanCn0",
            "MeanUnc",
            "SatCount",
            "WlsLat",
            "WlsLon",
        ]

        # Sanitize data to prevent NaN propagation (Cite debug_lesson_2)
        if mode != "test":
            # Fix: Also drop rows where Ground Truth targets are missing to prevent NaN loss
            target_cols = ["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]
            # Only check targets that actually exist in the dataframe (safety)
            actual_target_cols = [c for c in target_cols if c in df_merged.columns]
            cols_to_check = feature_cols + actual_target_cols
            df_merged.dropna(subset=cols_to_check, inplace=True)
        else:
            df_merged[feature_cols] = df_merged[feature_cols].fillna(0)

        if df_merged.empty:
            continue

        X_trip = df_merged[feature_cols].values.astype(np.float32)

        # Windowing
        X_windowed = create_windows(X_trip, window_size)

        all_X.append(X_windowed)
        all_meta.append(df_merged[["tripId", "UnixTimeMillis", "WlsLat", "WlsLon"]])

        # Compute Targets (ENU Residuals) if train/val
        if mode != "test":
            # GT Lat/Lon/Alt (Alt from GT if available, else 0 or WLS)
            gt_lat = df_merged["LatitudeDegrees"].values
            gt_lon = df_merged["LongitudeDegrees"].values
            gt_alt = df_merged["AltitudeMeters"].values

            # WLS Lat/Lon/Alt
            base_lat = df_merged["WlsLat"].values
            base_lon = df_merged["WlsLon"].values
            base_alt = df_merged["WlsAlt"].values

            # Convert both to ECEF
            gt_x, gt_y, gt_z = WGS84_to_ECEF(gt_lat, gt_lon, gt_alt)
            base_x, base_y, base_z = WGS84_to_ECEF(base_lat, base_lon, base_alt)

            # Convert GT ECEF to ENU relative to WLS position
            # This gives the correction vector needed to go from WLS to GT
            e, n, u = ECEF_to_ENU(gt_x, gt_y, gt_z, base_lat, base_lon, base_alt)

            # Target: [DeltaEast, DeltaNorth]
            y_trip = np.stack([e, n], axis=1).astype(np.float32)
            all_y.append(y_trip)

    # Concatenate
    X_final = np.concatenate(all_X, axis=0)
    meta_final = pd.concat(all_meta, ignore_index=True)

    if mode != "test":
        y_final = np.concatenate(all_y, axis=0)
    else:
        y_final = None

    # Normalization
    # We flatten X to (N*W, F) to fit scaler, then reshape back
    N, W, F = X_final.shape
    X_flat = X_final.reshape(-1, F)

    if mode == "train":
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_flat)
        # Save scaler stats
        stats = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        with open(scaler_path, "w") as f:
            json.dump(stats, f)
    else:
        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler stats not found. Run training first.")
        with open(scaler_path, "r") as f:
            stats = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(stats["mean"])
        scaler.scale_ = np.array(stats["scale"])
        X_norm = scaler.transform(X_flat)

    X_final = X_norm.reshape(N, W, F)

    # Save to cache
    np.save(X_path, X_final)
    meta_final.to_parquet(meta_path)
    if y_final is not None:
        np.save(y_path, y_final)

    return X_final, y_final, meta_final


# -------------------------------------------------------------------
# 3. Training Function
# -------------------------------------------------------------------


def train_model(epochs=20, batch_size=256, learning_rate=1e-3, patience=5):
    print("Starting training pipeline...")

    # Load data
    X_train, y_train, _ = process_data("train")
    X_val, y_val, _ = process_data("val")

    # Create Datasets
    train_dataset = GNSSWindowDataset(X_train, y_train)
    val_dataset = GNSSWindowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Model Setup
    # Features: ['dLat', 'dLon', 'dAlt', 'MeanCn0', 'MeanUnc', 'SatCount', 'WlsLat', 'WlsLon']
    # Dynamic: First 6. Static: Last 2.
    model = DSTResNet(
        dynamic_features=6,
        static_features=2,
        window_size=X_train.shape[1],
        hidden_dim=128,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataset)

        print(
            f"Epoch {epoch+1}/{epochs} - Train MAE: {train_loss:.6f} - Val MAE: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./working/idea_3/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Validation MAE: {best_val_loss:.6f}")


# -------------------------------------------------------------------
# 4. Inference Function
# -------------------------------------------------------------------


def generate_submission(batch_size=256):
    print("Generating submission...")

    # Load Test Data
    X_test, _, meta_test = process_data("test")
    test_dataset = GNSSWindowDataset(X_test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Load Model
    model = DSTResNet(
        dynamic_features=6,
        static_features=2,
        window_size=X_test.shape[1],
        hidden_dim=128,
    )

    model_path = "./working/idea_3/best_model.pth"
    if not os.path.exists(model_path):
        print("Model file not found. Skipping inference.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds_list = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())

    # Predicted residuals (East, North) in meters
    preds_enu = np.concatenate(preds_list, axis=0)

    # Reconstruct Lat/Lon
    # We need the baseline WLS positions from meta_test
    wls_lat = meta_test["WlsLat"].values
    wls_lon = meta_test["WlsLon"].values
    # Assume 0 altitude for reconstruction reference if not available, or use WGS84 ellipsoid surface
    # ECEF_to_ENU requires a reference. We used WLS as reference for target generation.
    # So we inverse: Target = ENU(GT) relative to WLS.
    # Pred = ENU_pred.
    # Pred_ECEF = ENU_to_ECEF(Pred_E, Pred_N, 0, WlsLat, WlsLon, WlsAlt)
    # But we don't have WlsAlt in meta_test easily unless we saved it.
    # Wait, process_data saves WlsLat/WlsLon in meta. It doesn't save WlsAlt in meta df.
    # Let's assume WlsAlt=0 for the ENU conversion base, as horizontal error is dominant.
    # Or better, we can approximate degrees change:
    # dLat_deg = dNorth / 111320
    # dLon_deg = dEast / (111320 * cos(lat))

    d_east = preds_enu[:, 0]
    d_north = preds_enu[:, 1]

    # Approximation
    # 1 deg lat ~ 111132.954 - 559.822 cos(2lat) + 1.175 cos(4lat)
    # 1 deg lon ~ 111412.84 cos(lat) - 93.5 cos(3lat)
    # Simple spherical approximation is sufficient for small residuals (<100m)
    r_earth = 6378137.0
    d_lat_rad = d_north / r_earth
    d_lon_rad = d_east / (r_earth * np.cos(np.radians(wls_lat)))

    pred_lat = wls_lat + np.degrees(d_lat_rad)
    pred_lon = wls_lon + np.degrees(d_lon_rad)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": meta_test["tripId"],
            "UnixTimeMillis": meta_test["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    os.makedirs("./submission", exist_ok=True)
    submission.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
