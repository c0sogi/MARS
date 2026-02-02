import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

from library.config import Config
from library.utils import ecef_to_lla, latlon_to_meters, seed_everything

# Set random seed
seed_everything(Config.RANDOM_STATE)


class GNSSDataset(Dataset):
    """
    PyTorch Dataset for the Sky-Contextualized Sequence-to-Point model.
    """

    def __init__(self, traj_features, sky_features, targets=None):
        """
        Args:
            traj_features (np.ndarray): Shape (N, Window_Size, Channels)
            sky_features (np.ndarray): Shape (N, Context_Dim)
            targets (np.ndarray, optional): Shape (N, 2) for d_east, d_north.
        """
        self.traj_features = torch.FloatTensor(traj_features)
        # Permute to (N, Channels, Window_Size) for PyTorch Conv1d
        self.traj_features = self.traj_features.permute(0, 2, 1)

        self.sky_features = torch.FloatTensor(sky_features)

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.traj_features)

    def __getitem__(self, idx):
        traj = self.traj_features[idx]
        sky = self.sky_features[idx]

        if self.targets is not None:
            target = self.targets[idx]
            return traj, sky, target
        else:
            return traj, sky


def compute_sky_context(df_epoch):
    """
    Aggregates satellite statistics for a single epoch.
    """
    # df_epoch contains rows for all satellites in one timestamp
    stats = {}

    # Elevation
    elevs = df_epoch["SvElevationDegrees"].values
    stats["sky_mean_elev"] = np.nanmean(elevs) if len(elevs) > 0 else 0.0
    stats["sky_std_elev"] = np.nanstd(elevs) if len(elevs) > 0 else 0.0

    # Azimuth - Use vector averaging for circular mean, but simple std for spread proxy
    azimuths = np.radians(df_epoch["SvAzimuthDegrees"].values)
    if len(azimuths) > 0:
        # Mean vector
        sa = np.nanmean(np.sin(azimuths))
        ca = np.nanmean(np.cos(azimuths))
        mean_az = np.degrees(np.arctan2(sa, ca))
        stats["sky_mean_azimuth"] = mean_az if mean_az >= 0 else mean_az + 360.0
        # Spread (std dev of degrees)
        stats["sky_std_azimuth"] = np.nanstd(df_epoch["SvAzimuthDegrees"].values)
    else:
        stats["sky_mean_azimuth"] = 0.0
        stats["sky_std_azimuth"] = 0.0

    # Cn0
    cn0s = df_epoch["Cn0DbHz"].values
    stats["sky_mean_cn0"] = np.nanmean(cn0s) if len(cn0s) > 0 else 0.0
    stats["sky_std_cn0"] = np.nanstd(cn0s) if len(cn0s) > 0 else 0.0

    # Count
    stats["sat_count"] = len(df_epoch)

    return stats


def process_trip(trip_id, meta_row, is_train=True):
    """
    Processes a single trip: loads GNSS, merges with GT/Submission, creates windows.
    """
    # Load GNSS Data
    gnss_path = os.path.join(Config.INPUT_DIR, meta_row["gnss_path"])
    if not os.path.exists(gnss_path):
        return None, None, None, None

    # Read specific columns to save memory
    df_gnss = pd.read_csv(gnss_path, usecols=Config.GNSS_RAW_COLS)

    # Aggregate GNSS by epoch
    # 1. Get WLS Position (take first, as it is repeated per epoch)
    # 2. Get Mean Signal Metrics
    # 3. Get Sky Statistics

    # Group by timestamp
    grouped = df_gnss.groupby("utcTimeMillis")

    epoch_data = []

    for time_millis, group in grouped:
        # Basic epoch info
        first_row = group.iloc[0]

        # WLS Position ECEF
        wls_x = first_row["WlsPositionXEcefMeters"]
        wls_y = first_row["WlsPositionYEcefMeters"]
        wls_z = first_row["WlsPositionZEcefMeters"]

        # Signal Metrics
        mean_cn0 = group["Cn0DbHz"].mean()
        mean_unc = group["RawPseudorangeUncertaintyMeters"].mean()

        # Sky Context
        sky_ctx = compute_sky_context(group)

        epoch_record = {
            "utcTimeMillis": time_millis,
            "wls_x": wls_x,
            "wls_y": wls_y,
            "wls_z": wls_z,
            "mean_cn0": mean_cn0,
            "mean_uncertainty": mean_unc,
            **sky_ctx,
        }
        epoch_data.append(epoch_record)

    df_epochs = pd.DataFrame(epoch_data)

    # Convert WLS ECEF to LLA
    lat, lon, alt = ecef_to_lla(
        df_epochs["wls_x"].values, df_epochs["wls_y"].values, df_epochs["wls_z"].values
    )
    df_epochs["wls_lat"] = lat
    df_epochs["wls_lon"] = lon
    df_epochs["wls_alt"] = alt

    # Sort by time
    df_epochs = df_epochs.sort_values("utcTimeMillis").reset_index(drop=True)

    # Merge with Ground Truth or Sample Submission
    # Train/Val: merge with Ground Truth on UnixTimeMillis
    # Test: merge with Sample Submission on UnixTimeMillis

    if is_train:
        # Load Ground Truth from metadata row (it's not in a separate file for the row,
        # but we need the full GT for this trip to merge)
        # Actually, the metadata CSV passed to this function usually contains one row per trip
        # if we grouped it, but here we likely iterate over unique trips from the metadata dataframe.
        # We need to filter the main metadata df for this trip.
        pass  # Logic handled in batch processing below

    return df_epochs


def create_windows(df_merged, window_size):
    """
    Creates sliding windows from the merged dataframe.
    Returns lists of trajectory arrays, sky arrays, target arrays, and metadata.
    """
    traj_list = []
    sky_list = []
    target_list = []
    meta_list = []

    # Ensure sorted
    df_merged = df_merged.sort_values("utcTimeMillis").reset_index(drop=True)

    # We need continuous data. Check time diffs.
    # Timestamps are in millis. 1 sec = 1000 ms.
    time_diffs = df_merged["utcTimeMillis"].diff()

    # Identify breaks (gap > 1.5s)
    breaks = np.where(time_diffs > 1500)[0]

    # Create segments
    segments = []
    start_idx = 0
    for b_idx in breaks:
        segments.append(df_merged.iloc[start_idx:b_idx])
        start_idx = b_idx
    segments.append(df_merged.iloc[start_idx:])

    half_window = window_size // 2

    for seg in segments:
        if len(seg) < window_size:
            continue

        # Convert to numpy for speed
        # Features needed for trajectory: lat, lon, alt, mean_cn0, mean_uncertainty
        # We calculate relative coords and velocity on the fly per window

        # Extract columns
        # WLS LLA
        wls_pos = seg[["wls_lat", "wls_lon", "wls_alt"]].values
        # Signal
        sig_metrics = seg[["mean_cn0", "mean_uncertainty"]].values
        # Sky Context (we average this over the window later, or take center?)
        # Config says: "Aggregated statistics over the satellites visible in the window"
        # We have per-epoch stats. We can average these over the window.
        sky_cols = [
            c for c in Config.CONTEXT_FEATURES if c != "sat_count"
        ]  # sat_count is also averaged
        sky_cols = Config.CONTEXT_FEATURES  # All of them
        sky_data = seg[sky_cols].values

        # Targets (if available)
        has_target = "d_east" in seg.columns
        if has_target:
            targets = seg[["d_east", "d_north"]].values

        # Metadata
        meta_cols = ["tripId", "utcTimeMillis", "wls_lat", "wls_lon"]
        meta_data = seg[meta_cols].values

        num_samples = len(seg) - window_size + 1

        for i in range(num_samples):
            # Window indices
            start = i
            end = i + window_size
            center = i + half_window

            # --- Trajectory Features ---
            # 1. Relative Position
            center_lat = wls_pos[center, 0]
            center_lon = wls_pos[center, 1]
            center_alt = wls_pos[center, 2]

            window_lat = wls_pos[start:end, 0]
            window_lon = wls_pos[start:end, 1]
            window_alt = wls_pos[start:end, 2]

            # Convert deg to meters relative to center
            # lat diff in m
            rel_lat_m = (window_lat - center_lat) * 111320.0
            # lon diff in m
            rel_lon_m = (
                (window_lon - center_lon) * 111320.0 * np.cos(np.radians(center_lat))
            )
            # alt diff
            rel_alt_m = window_alt - center_alt

            # 2. Velocity (First order difference)
            # Pad first element with 0 or repeat? Let's use gradient.
            vel_lat_m = np.gradient(rel_lat_m)
            vel_lon_m = np.gradient(rel_lon_m)
            vel_alt_m = np.gradient(rel_alt_m)

            # 3. Signal Metrics
            window_sig = sig_metrics[start:end]

            # Stack trajectory features: (Window, Channels)
            # Channels: rel_lat, rel_lon, rel_alt, vel_lat, vel_lon, vel_alt, cn0, unc
            traj_feat = np.stack(
                [
                    rel_lat_m,
                    rel_lon_m,
                    rel_alt_m,
                    vel_lat_m,
                    vel_lon_m,
                    vel_alt_m,
                    window_sig[:, 0],
                    window_sig[:, 1],
                ],
                axis=1,
            )

            # --- Sky Context Features ---
            # Average sky stats over the window
            window_sky = np.mean(sky_data[start:end], axis=0)

            # --- Target ---
            if has_target:
                target_val = targets[center]
                target_list.append(target_val)

            # --- Metadata ---
            # We store metadata for the center frame
            meta_list.append(meta_data[center])

            traj_list.append(traj_feat)
            sky_list.append(window_sky)

    return traj_list, sky_list, target_list, meta_list


def process_dataset(metadata_path, is_train=True):
    """
    Loads metadata, processes all trips, and returns concatenated arrays.
    """
    df_meta = pd.read_csv(metadata_path)
    unique_trips = df_meta["tripId"].unique()

    all_traj = []
    all_sky = []
    all_targets = []
    all_meta = []

    print(f"Processing {len(unique_trips)} trips from {metadata_path}...")

    for trip_id in tqdm(unique_trips):
        # Filter metadata for this trip
        trip_subset = df_meta[df_meta["tripId"] == trip_id].copy()

        # Get one row to find file paths
        row = trip_subset.iloc[0]

        # Process GNSS for this trip
        df_gnss_epochs = process_trip(trip_id, row, is_train)

        if df_gnss_epochs is None or df_gnss_epochs.empty:
            continue

        # Merge with target/submission timestamps
        # For Train/Val, we have Ground Truth in trip_subset
        # For Test, we have required timestamps in trip_subset

        # Rename for merge
        trip_subset = trip_subset.rename(columns={"UnixTimeMillis": "utcTimeMillis"})

        # Merge
        # We use inner join to ensure we only keep epochs where we have GT (or required output)
        df_merged = pd.merge(
            df_gnss_epochs, trip_subset, on=["tripId", "utcTimeMillis"], how="inner"
        )

        if df_merged.empty:
            continue

        # If train, calculate residuals (Targets)
        if is_train:
            # Ground Truth is LatitudeDegrees, LongitudeDegrees
            # Baseline is wls_lat, wls_lon
            # Calculate d_east, d_north
            d_east, d_north = latlon_to_meters(
                df_merged["wls_lat"].values,
                df_merged["wls_lon"].values,
                df_merged["LatitudeDegrees"].values,
                df_merged["LongitudeDegrees"].values,
            )
            df_merged["d_east"] = d_east
            df_merged["d_north"] = d_north

        # Create windows
        traj, sky, tgt, meta = create_windows(df_merged, Config.WINDOW_SIZE)

        all_traj.extend(traj)
        all_sky.extend(sky)
        all_targets.extend(tgt)
        all_meta.extend(meta)

    # Convert to numpy arrays
    X_traj = np.array(all_traj, dtype=np.float32)
    X_sky = np.array(all_sky, dtype=np.float32)
    y = (
        np.array(all_targets, dtype=np.float32)
        if all_targets
        else np.zeros((len(X_traj), 0))
    )

    # Meta dataframe
    df_meta_out = pd.DataFrame(
        all_meta, columns=["tripId", "utcTimeMillis", "wls_lat", "wls_lon"]
    )

    return X_traj, X_sky, y, df_meta_out


def load_and_cache_data(
    metadata_path, cache_path, is_train=True, load_cached_data=True
):
    """
    Loads data from cache or processes it from scratch.
    """
    # Define cache filenames
    base_name = os.path.splitext(os.path.basename(cache_path))[0]
    traj_path = os.path.join(Config.WORKING_DIR, f"{base_name}_traj.npy")
    sky_path = os.path.join(Config.WORKING_DIR, f"{base_name}_sky.npy")
    y_path = os.path.join(Config.WORKING_DIR, f"{base_name}_y.npy")
    meta_path = os.path.join(Config.WORKING_DIR, f"{base_name}_meta.parquet")

    if load_cached_data and os.path.exists(traj_path) and os.path.exists(meta_path):
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        X_traj = np.load(traj_path)
        X_sky = np.load(sky_path)
        y = np.load(y_path)
        df_meta = pd.read_parquet(meta_path)
        return X_traj, X_sky, y, df_meta

    print(f"Processing data from scratch for {metadata_path}...")
    X_traj, X_sky, y, df_meta = process_dataset(metadata_path, is_train)

    # Save to cache
    print(f"Saving data to {Config.WORKING_DIR}...")
    np.save(traj_path, X_traj)
    np.save(sky_path, X_sky)
    np.save(y_path, y)
    df_meta.to_parquet(meta_path)

    return X_traj, X_sky, y, df_meta


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # 1. Load Train
    X_traj_train, X_sky_train, y_train, _ = load_and_cache_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        is_train=True,
        load_cached_data=load_cached_data,
    )

    # 2. Load Val
    X_traj_val, X_sky_val, y_val, val_meta = load_and_cache_data(
        Config.VAL_METADATA_PATH,
        Config.VAL_CACHE_PATH,
        is_train=True,
        load_cached_data=load_cached_data,
    )

    # 3. Load Test
    X_traj_test, X_sky_test, _, test_meta = load_and_cache_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        is_train=False,
        load_cached_data=load_cached_data,
    )

    # 4. Scaling
    # We need to scale Trajectory features and Sky features separately.
    # Trajectory: (N, Window, Channels). Reshape to (N*Window, Channels) for scaling.

    print("Fitting Scalers...")

    # Trajectory Scaler
    N, W, C = X_traj_train.shape
    traj_scaler = StandardScaler()
    X_traj_train_flat = X_traj_train.reshape(-1, C)
    X_traj_train_flat = traj_scaler.fit_transform(X_traj_train_flat)
    X_traj_train = X_traj_train_flat.reshape(N, W, C)

    # Apply to Val
    N_val, W_val, C_val = X_traj_val.shape
    X_traj_val = traj_scaler.transform(X_traj_val.reshape(-1, C_val)).reshape(
        N_val, W_val, C_val
    )

    # Apply to Test
    N_test, W_test, C_test = X_traj_test.shape
    X_traj_test = traj_scaler.transform(X_traj_test.reshape(-1, C_test)).reshape(
        N_test, W_test, C_test
    )

    # Sky Context Scaler
    sky_scaler = StandardScaler()
    X_sky_train = sky_scaler.fit_transform(X_sky_train)
    X_sky_val = sky_scaler.transform(X_sky_val)
    X_sky_test = sky_scaler.transform(X_sky_test)

    # Save scalers
    scalers = {"traj": traj_scaler, "sky": sky_scaler}
    joblib.dump(scalers, Config.SCALER_PATH)

    # 5. Create Datasets
    train_dataset = GNSSDataset(X_traj_train, X_sky_train, y_train)
    val_dataset = GNSSDataset(X_traj_val, X_sky_val, y_val)
    test_dataset = GNSSDataset(X_traj_test, X_sky_test, None)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, val_meta, test_meta
