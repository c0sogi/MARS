import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import get_logger, ecef_to_lla, lla_to_enu

# Initialize logger
logger = get_logger("data_loader")


class SmartphoneGNSSDataset(Dataset):
    """
    PyTorch Dataset for Smartphone GNSS Data.
    Yields (sequence_input, target_residual) pairs.
    """

    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_scaler(scaler_path):
    """
    Loads or creates a StandardScaler.
    """
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "r") as f:
                data = json.load(f)
            scaler = StandardScaler()
            scaler.mean_ = np.array(data["mean"])
            scaler.scale_ = np.array(data["scale"])
            scaler.var_ = np.array(data["var"])
            scaler.n_samples_seen_ = data["n_samples_seen"]
            return scaler
        except Exception as e:
            logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
            return StandardScaler()
    else:
        return StandardScaler()


def save_scaler(scaler, scaler_path):
    """
    Saves StandardScaler parameters to JSON.
    """
    data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }
    with open(scaler_path, "w") as f:
        json.dump(data, f)


def process_trip(trip_id, trip_meta, gnss_df, is_train=True):
    """
    Process a single trip: alignment, WLS baseline, ENU conversion, windowing.
    """
    # 1. Aggregate GNSS data by epoch
    # Rename columns for consistency
    gnss_df = gnss_df.rename(columns=Config.AGG_RENAME)

    # Group by utcTimeMillis
    # We use the aggregation rules from Config
    # Filter agg_cols to only those present in df
    agg_rules = {k: v for k, v in Config.AGG_COLS.items() if k in gnss_df.columns}

    df_gnss_agg = gnss_df.groupby("utcTimeMillis").agg(agg_rules).reset_index()

    # Rename aggregated columns again if grouping messed them up or if they weren't renamed before
    # (The groupby might keep original names, so we apply rename map again just in case)
    df_gnss_agg = df_gnss_agg.rename(columns=Config.AGG_RENAME)

    # 2. Align with Target Timestamps
    # The trip_meta contains the timestamps we MUST predict for (or train on)
    target_timestamps = trip_meta[["tripId", "UnixTimeMillis"]].copy()

    # Merge GNSS data onto target timestamps
    # We use merge_asof or simple merge. Since description says "expected timestamps",
    # and we aggregated by utcTimeMillis, exact merge is preferred.
    # However, to be robust against slight misalignments, we can use nearest with a tolerance.
    # But usually, the provided GT and GNSS logs are aligned.

    df_merged = pd.merge(
        target_timestamps,
        df_gnss_agg,
        left_on="UnixTimeMillis",
        right_on="utcTimeMillis",
        how="left",
    )

    # Fill missing GNSS data (if any gaps)
    # Forward fill then backward fill to handle gaps
    cols_to_fill = [
        c for c in df_merged.columns if c not in ["UnixTimeMillis", "utcTimeMillis"]
    ]
    df_merged[cols_to_fill] = df_merged[cols_to_fill].ffill().bfill()

    # If still NaNs (e.g. empty GNSS file), fill with 0 (should be rare/handled by filtering)
    df_merged[cols_to_fill] = df_merged[cols_to_fill].fillna(0)

    # 3. Compute Baseline WLS LLA
    x = df_merged["WlsPositionXEcefMeters"].values
    y = df_merged["WlsPositionYEcefMeters"].values
    z = df_merged["WlsPositionZEcefMeters"].values

    wls_lat, wls_lon, wls_alt = ecef_to_lla(x, y, z)

    # 4. Convert to Trip-Relative ENU
    # Reference point: First valid WLS position of the trip
    lat0, lon0, alt0 = wls_lat[0], wls_lon[0], wls_alt[0]

    # WLS ENU
    e_wls, n_wls, u_wls = lla_to_enu(wls_lat, wls_lon, wls_alt, lat0, lon0, alt0)

    # 5. Compute Targets (if training)
    targets = None
    if is_train:
        gt_lat = trip_meta["LatitudeDegrees"].values
        gt_lon = trip_meta["LongitudeDegrees"].values
        gt_alt = trip_meta["AltitudeMeters"].values

        e_gt, n_gt, u_gt = lla_to_enu(gt_lat, gt_lon, gt_alt, lat0, lon0, alt0)

        # Target is residual: GT - WLS
        target_e = e_gt - e_wls
        target_n = n_gt - n_wls
        targets = np.stack([target_e, target_n], axis=1)

    # 6. Construct Features for Windowing
    # Features: [East, North, Up, Cn0, Unc, Sat]
    # We will handle "Relative Centering" and "Deltas" after creating windows to support vectorization

    # Base features array: (T, C_base)
    # C_base = 3 (Pos) + 3 (Signal)
    pos_feats = np.stack([e_wls, n_wls, u_wls], axis=1)
    sig_feats = df_merged[["MeanCn0", "MeanUncertainty", "SatCount"]].values

    # Combine
    base_data = np.concatenate([pos_feats, sig_feats], axis=1)

    # 7. Create Sliding Windows
    # We need a window for EVERY timestamp in df_merged.
    # We pad the beginning and end to maintain length T.
    window_size = Config.WINDOW_SIZE
    pad_size = window_size // 2

    # Pad with edge values
    data_padded = np.pad(base_data, ((pad_size, pad_size), (0, 0)), mode="edge")

    # Create strided view
    # Shape: (T, Window_Size, Features)
    # Using numpy stride tricks for efficiency
    # stride_tricks.sliding_window_view is available in numpy >= 1.20
    windows = np.lib.stride_tricks.sliding_window_view(data_padded, window_size, axis=0)
    # Current shape: (T, Features, Window_Size). Transpose to (T, Window_Size, Features)
    windows = windows.transpose(0, 2, 1)

    # 8. Feature Engineering on Windows
    # windows shape: (T, W, 6) where 6 is [E, N, U, Cn0, Unc, Sat]

    # 8a. Relative Centering (Local Shape)
    # Center index in window is pad_size
    # Extract center position: (T, 1, 3)
    center_pos = windows[:, pad_size : pad_size + 1, 0:3]

    # Calculate relative position: Window_Pos - Center_Pos
    rel_pos = windows[:, :, 0:3] - center_pos

    # 8b. Dynamics (Deltas)
    # Calculate diff along time axis (axis 1)
    # Prepend 0 to keep size W
    # diffs shape: (T, W-1, 3)
    pos_window = windows[:, :, 0:3]
    diffs = np.diff(pos_window, axis=1)
    # Pad the first delta with 0s to match window size
    zeros = np.zeros((diffs.shape[0], 1, 3))
    deltas = np.concatenate([zeros, diffs], axis=1)

    # 8c. Signal Features
    signals = windows[:, :, 3:6]

    # Concatenate all: [Rel_Pos(3), Deltas(3), Signals(3)]
    # Final Feature Order: rel_e, rel_n, rel_u, d_e, d_n, d_u, cn0, unc, sat
    X_trip = np.concatenate([rel_pos, deltas, signals], axis=2)

    return X_trip, targets, df_merged[["tripId", "UnixTimeMillis"]]


def load_and_preprocess_data(split="train", debug=False, load_cached_data=True):
    """
    Main function to load and preprocess data.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, use a subset of data.
        load_cached_data (bool): If True, attempt to load from cache.

    Returns:
        dataset (SmartphoneGNSSDataset): PyTorch dataset.
        metadata (pd.DataFrame): Metadata corresponding to the dataset (for inference).
    """
    cache_file_X = os.path.join(Config.CACHE_DIR, f"{split}_X.npy")
    cache_file_y = os.path.join(Config.CACHE_DIR, f"{split}_y.npy")
    cache_file_meta = os.path.join(Config.CACHE_DIR, f"{split}_meta.parquet")
    scaler_path = os.path.join(Config.CACHE_DIR, "scaler_stats.json")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_file_X)
        and os.path.exists(cache_file_meta)
    ):
        logger.info(f"Loading {split} data from cache...")
        X = np.load(cache_file_X)
        meta = pd.read_parquet(cache_file_meta)
        y = np.load(cache_file_y) if os.path.exists(cache_file_y) else None

        return SmartphoneGNSSDataset(X, y), meta

    # 2. Load Metadata
    logger.info(f"Processing {split} data from scratch...")
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    df_meta = pd.read_csv(meta_path)

    if debug:
        logger.info(f"Debug mode: sampling {Config.SAMPLE_SIZE} trips.")
        unique_trips = df_meta["tripId"].unique()
        if len(unique_trips) > Config.SAMPLE_SIZE:
            sampled_trips = np.random.choice(
                unique_trips, Config.SAMPLE_SIZE, replace=False
            )
            df_meta = df_meta[df_meta["tripId"].isin(sampled_trips)].copy()

    # 3. Process Trips
    X_list = []
    y_list = []
    meta_list = []

    unique_trips = df_meta["tripId"].unique()

    for trip_id in unique_trips:
        # Get metadata for this trip
        trip_meta_rows = df_meta[df_meta["tripId"] == trip_id].sort_values(
            "UnixTimeMillis"
        )

        # Get GNSS file path
        gnss_rel_path = trip_meta_rows.iloc[0]["gnss_path"]
        gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            logger.warning(f"GNSS file not found: {gnss_path}")
            continue

        # Load GNSS Raw
        gnss_df = pd.read_csv(gnss_path, usecols=Config.RAW_GNSS_COLS)

        # Process
        X_trip, y_trip, meta_trip = process_trip(
            trip_id, trip_meta_rows, gnss_df, is_train=(split != "test")
        )

        X_list.append(X_trip)
        meta_list.append(meta_trip)
        if y_trip is not None:
            y_list.append(y_trip)

    # Concatenate
    X = np.concatenate(X_list, axis=0)
    meta = pd.concat(meta_list, ignore_index=True)
    y = np.concatenate(y_list, axis=0) if y_list else None

    # 4. Scaling
    # We scale features: (N, W, C) -> reshape to (N*W, C) -> scale -> reshape back
    N, W, C = X.shape
    X_reshaped = X.reshape(-1, C)

    if split == "train":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        save_scaler(scaler, scaler_path)
        logger.info("Scaler fitted and saved.")
    else:
        scaler = get_scaler(scaler_path)
        X_scaled = scaler.transform(X_reshaped)

    X = X_scaled.reshape(N, W, C)

    # 5. Save to Cache
    logger.info(f"Saving {split} data to cache...")
    np.save(cache_file_X, X)
    meta.to_parquet(cache_file_meta)
    if y is not None:
        np.save(cache_file_y, y)

    return SmartphoneGNSSDataset(X, y), meta
