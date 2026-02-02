import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import hashlib
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WINDOW_SIZE,
    WINDOW_CENTER_IDX,
    TRAJ_FEATURES,
    SKY_FEATURES,
    LAT_METERS_PER_DEGREE,
)
from library.utils import wgs84_to_meters_relative


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.
    Vectorized implementation.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Altitude (approximate)
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


class GNSSWindowDataset(Dataset):
    def __init__(self, traj_feats, sky_feats, targets=None, meta=None):
        self.traj_feats = torch.FloatTensor(traj_feats)
        self.sky_feats = torch.FloatTensor(sky_feats)

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

        self.meta = meta

    def __len__(self):
        return len(self.traj_feats)

    def __getitem__(self, idx):
        sample = {"traj_feat": self.traj_feats[idx], "sky_feat": self.sky_feats[idx]}

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        if self.meta is not None:
            sample["meta"] = self.meta[idx]  # trip_id, timestamp, wls_lat, wls_lon

        return sample


def preprocess_trip(trip_id, trip_meta_df, gnss_path, imu_path, load_cached_data=True):
    """
    Process a single trip: load raw data, align timestamps, create windows,
    compute relative features and sky context.
    """
    # Create a hash for caching based on trip ID and configuration
    config_str = f"{trip_id}_{WINDOW_SIZE}_{TRAJ_FEATURES}_{SKY_FEATURES}"
    trip_hash = hashlib.md5(config_str.encode()).hexdigest()

    cache_file_traj = os.path.join(CACHE_DIR, f"{trip_hash}_traj.npy")
    cache_file_sky = os.path.join(CACHE_DIR, f"{trip_hash}_sky.npy")
    cache_file_y = os.path.join(CACHE_DIR, f"{trip_hash}_y.npy")
    cache_file_meta = os.path.join(CACHE_DIR, f"{trip_hash}_meta.npy")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_file_traj)
            and os.path.exists(cache_file_sky)
            and os.path.exists(cache_file_y)
            and os.path.exists(cache_file_meta)
        ):
            return (
                np.load(cache_file_traj),
                np.load(cache_file_sky),
                np.load(cache_file_y),
                np.load(cache_file_meta, allow_pickle=True),
            )

    # --- Load Raw Data ---
    full_gnss_path = os.path.join(INPUT_DIR, gnss_path)
    full_imu_path = os.path.join(INPUT_DIR, imu_path)

    if not os.path.exists(full_gnss_path):
        # Return empty if file missing (should not happen with correct metadata)
        return np.array([]), np.array([]), np.array([]), np.array([])

    gnss_df = pd.read_csv(full_gnss_path)

    # --- Aggregate GNSS by Epoch ---
    # We need to aggregate satellite data to get one row per timestamp
    # Features to aggregate:
    # - WLS Position (take first, they should be same for same epoch)
    # - Cn0 (mean, std)
    # - Elevation/Azimuth (mean, std)
    # - Uncertainties (mean)
    # - Count of satellites

    # Define aggregation dictionary
    agg_dict = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "SvElevationDegrees": ["mean", "std"],
        "SvAzimuthDegrees": ["mean", "std"],
        "Cn0DbHz": ["mean", "std"],
        "RawPseudorangeUncertaintyMeters": "mean",
        "ReceivedSvTimeUncertaintyNanos": "mean",
        "Svid": "count",
    }

    # Filter for existing columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in gnss_df.columns}

    gnss_epoch = gnss_df.groupby("utcTimeMillis").agg(agg_dict)

    # Flatten columns
    gnss_epoch.columns = ["_".join(col).strip() for col in gnss_epoch.columns.values]
    gnss_epoch.reset_index(inplace=True)

    # Rename for clarity
    rename_map = {
        "WlsPositionXEcefMeters_first": "wls_x",
        "WlsPositionYEcefMeters_first": "wls_y",
        "WlsPositionZEcefMeters_first": "wls_z",
        "SvElevationDegrees_mean": "mean_elev",
        "SvElevationDegrees_std": "std_elev",
        "SvAzimuthDegrees_mean": "mean_azim",
        "SvAzimuthDegrees_std": "std_azim",
        "Cn0DbHz_mean": "mean_cn0",
        "Cn0DbHz_std": "std_cn0",
        "RawPseudorangeUncertaintyMeters_mean": "mean_pr_unc",
        "ReceivedSvTimeUncertaintyNanos_mean": "mean_sv_time_unc",
        "Svid_count": "sat_count",
    }
    gnss_epoch.rename(columns=rename_map, inplace=True)

    # Convert WLS ECEF to LLA
    wls_lat, wls_lon, wls_alt = ecef_to_lla(
        gnss_epoch["wls_x"].values,
        gnss_epoch["wls_y"].values,
        gnss_epoch["wls_z"].values,
    )
    gnss_epoch["wls_lat"] = wls_lat
    gnss_epoch["wls_lon"] = wls_lon
    gnss_epoch["wls_alt"] = wls_alt

    # Fill NaNs in std columns (single satellite case)
    gnss_epoch.fillna(0, inplace=True)

    # --- Process IMU Data ---
    if os.path.exists(full_imu_path):
        imu_df = pd.read_csv(full_imu_path)
        # Interpolate IMU to GNSS timestamps
        # Sort both
        gnss_epoch = gnss_epoch.sort_values("utcTimeMillis")
        imu_df = imu_df.sort_values("utcTimeMillis")

        t_gnss = gnss_epoch["utcTimeMillis"].values
        t_imu = imu_df["utcTimeMillis"].values

        # Linear interpolation for accel
        for axis in ["X", "Y", "Z"]:
            col = f"Measurement{axis}"
            if col in imu_df.columns:
                gnss_epoch[f"acc_{axis.lower()}"] = np.interp(
                    t_gnss, t_imu, imu_df[col].values
                )
            else:
                gnss_epoch[f"acc_{axis.lower()}"] = 0.0
    else:
        gnss_epoch["acc_x"] = 0.0
        gnss_epoch["acc_y"] = 0.0
        gnss_epoch["acc_z"] = 0.0

    # --- Calculate Velocity ---
    # Simple finite difference of WLS position
    # Convert lat/lon diff to meters for velocity calculation
    # Note: This is a rough estimate, but sufficient for relative dynamics
    d_lat = np.diff(gnss_epoch["wls_lat"], prepend=gnss_epoch["wls_lat"].iloc[0])
    d_lon = np.diff(gnss_epoch["wls_lon"], prepend=gnss_epoch["wls_lon"].iloc[0])
    d_alt = np.diff(gnss_epoch["wls_alt"], prepend=gnss_epoch["wls_alt"].iloc[0])
    d_t = (
        np.diff(
            gnss_epoch["utcTimeMillis"], prepend=gnss_epoch["utcTimeMillis"].iloc[0]
        )
        / 1000.0
    )
    d_t[d_t == 0] = 1.0  # Avoid div by zero

    # Meters
    vy = d_lat * LAT_METERS_PER_DEGREE
    vx = (
        d_lon * LAT_METERS_PER_DEGREE * np.cos(np.radians(gnss_epoch["wls_lat"].values))
    )

    gnss_epoch["vel_x"] = vx / d_t
    gnss_epoch["vel_y"] = vy / d_t
    gnss_epoch["vel_z"] = d_alt / d_t

    # --- Prepare Windows ---
    # We only generate windows for timestamps requested in trip_meta_df
    # trip_meta_df contains 'UnixTimeMillis' which corresponds to 'utcTimeMillis'

    target_timestamps = trip_meta_df["UnixTimeMillis"].values

    # Create lookup for GNSS data
    # We use searchsorted to find nearest indices, but we strictly want exact matches or very close
    # The problem statement says "Reference locations at expected timestamps".
    # We assume GNSS epochs exist for these timestamps.

    # Map timestamp to index in gnss_epoch
    time_to_idx = {t: i for i, t in enumerate(gnss_epoch["utcTimeMillis"].values)}

    X_traj_list = []
    X_sky_list = []
    y_list = []
    meta_list = []

    gnss_timestamps = gnss_epoch["utcTimeMillis"].values
    n_epochs = len(gnss_epoch)

    # Pre-extract columns to numpy for speed
    # Trajectory features per step
    traj_data = np.zeros((n_epochs, len(TRAJ_FEATURES)))
    # Placeholder for relative positions (calculated per window)
    # We store absolute WLS and other features here
    wls_pos = gnss_epoch[["wls_lat", "wls_lon", "wls_alt"]].values

    # Other features: vel, acc, cn0, unc
    # TRAJ_FEATURES = [rel_pos_x, rel_pos_y, rel_pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, mean_cn0, mean_pr_unc, mean_sv_time_unc]
    # Indices 0,1,2 are rel pos (computed later)
    traj_data[:, 3] = gnss_epoch["vel_x"].values
    traj_data[:, 4] = gnss_epoch["vel_y"].values
    traj_data[:, 5] = gnss_epoch["vel_z"].values
    traj_data[:, 6] = gnss_epoch["acc_x"].values
    traj_data[:, 7] = gnss_epoch["acc_y"].values
    traj_data[:, 8] = gnss_epoch["acc_z"].values
    traj_data[:, 9] = gnss_epoch["mean_cn0"].values
    traj_data[:, 10] = gnss_epoch["mean_pr_unc"].values
    traj_data[:, 11] = gnss_epoch["mean_sv_time_unc"].values

    # Sky features
    # SKY_FEATURES = [mean_elev, std_elev, mean_azim, std_azim, mean_cn0_sky, std_cn0_sky, sat_count_mean]
    sky_data_source = gnss_epoch[
        [
            "mean_elev",
            "std_elev",
            "mean_azim",
            "std_azim",
            "mean_cn0",
            "std_cn0",
            "sat_count",
        ]
    ].values

    # Ground Truth map
    gt_map = {}
    if "LatitudeDegrees" in trip_meta_df.columns:
        for _, row in trip_meta_df.iterrows():
            gt_map[row["UnixTimeMillis"]] = (
                row["LatitudeDegrees"],
                row["LongitudeDegrees"],
            )

    for target_ts in target_timestamps:
        # Find index in GNSS data
        # We use searchsorted to find insertion point
        idx = np.searchsorted(gnss_timestamps, target_ts)

        # Check if exact match or close enough (within 10ms?)
        # If not exact match, we might be extrapolating.
        # For this implementation, we take the nearest valid index.
        if idx >= n_epochs:
            idx = n_epochs - 1
        elif idx > 0:
            # Check which is closer
            if abs(gnss_timestamps[idx] - target_ts) > abs(
                gnss_timestamps[idx - 1] - target_ts
            ):
                idx = idx - 1

        # Define Window Indices
        start_idx = idx - WINDOW_CENTER_IDX
        end_idx = start_idx + WINDOW_SIZE

        indices = np.arange(start_idx, end_idx)
        # Clamp indices
        indices = np.clip(indices, 0, n_epochs - 1)

        # --- Trajectory Features ---
        # Get data for window
        window_traj = traj_data[indices].copy()
        window_wls = wls_pos[indices]

        # Center WLS (at target index)
        center_wls = wls_pos[idx]  # Lat, Lon, Alt

        # Calculate Relative Positions (Meters)
        # wgs84_to_meters_relative returns (x, y)
        dx, dy = wgs84_to_meters_relative(
            center_wls[0], center_wls[1], window_wls[:, 0], window_wls[:, 1]
        )
        dz = window_wls[:, 2] - center_wls[2]

        window_traj[:, 0] = dx
        window_traj[:, 1] = dy
        window_traj[:, 2] = dz

        # Flatten
        X_traj_list.append(window_traj.flatten())

        # --- Sky Context Features ---
        # Aggregate over window
        window_sky_raw = sky_data_source[indices]
        # Mean of means, mean of stds, etc.
        # SKY_FEATURES = [mean_elev, std_elev, mean_azim, std_azim, mean_cn0_sky, std_cn0_sky, sat_count_mean]
        sky_feat_vec = np.mean(window_sky_raw, axis=0)
        X_sky_list.append(sky_feat_vec)

        # --- Target ---
        if target_ts in gt_map:
            gt_lat, gt_lon = gt_map[target_ts]
            # Target is offset from WLS center to GT
            t_dx, t_dy = wgs84_to_meters_relative(
                center_wls[0], center_wls[1], gt_lat, gt_lon
            )
            y_list.append([t_dx, t_dy])
        else:
            y_list.append([np.nan, np.nan])  # Should not happen for train/val

        # --- Meta ---
        meta_list.append([trip_id, target_ts, center_wls[0], center_wls[1]])

    # Convert to numpy
    X_traj = np.array(X_traj_list, dtype=np.float32)
    X_sky = np.array(X_sky_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = np.array(meta_list, dtype=object)

    # Save to cache
    np.save(cache_file_traj, X_traj)
    np.save(cache_file_sky, X_sky)
    np.save(cache_file_y, y)
    np.save(cache_file_meta, meta)

    return X_traj, X_sky, y, meta


def load_dataset(mode="train", scaler=None, max_samples=None, load_cached_data=True):
    """
    Main function to load dataset.

    Args:
        mode: 'train', 'val', or 'test'
        scaler: sklearn StandardScaler (optional, required for val/test)
        max_samples: Limit number of samples (for debugging)
        load_cached_data: Whether to use cached numpy files

    Returns:
        dataset: GNSSWindowDataset
        scaler: Fitted scaler (if mode='train')
    """
    if mode == "train":
        meta_path = TRAIN_METADATA_PATH
    elif mode == "val":
        meta_path = VAL_METADATA_PATH
    else:
        meta_path = TEST_METADATA_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found.")

    df_meta = pd.read_csv(meta_path)

    # Group by trip
    trips = df_meta["tripId"].unique()

    all_X_traj = []
    all_X_sky = []
    all_y = []
    all_meta = []

    print(f"Loading {mode} data from {len(trips)} trips...")

    for trip_id in tqdm(trips):
        trip_data = df_meta[df_meta["tripId"] == trip_id]
        # Get paths from the first row of the trip
        gnss_path = trip_data.iloc[0]["gnss_path"]
        imu_path = trip_data.iloc[0]["imu_path"]

        X_t, X_s, y, m = preprocess_trip(
            trip_id, trip_data, gnss_path, imu_path, load_cached_data
        )

        if len(X_t) > 0:
            all_X_traj.append(X_t)
            all_X_sky.append(X_s)
            all_y.append(y)
            all_meta.append(m)

        if max_samples and sum(len(x) for x in all_X_traj) >= max_samples:
            break

    if not all_X_traj:
        raise ValueError("No data loaded.")

    X_traj = np.concatenate(all_X_traj, axis=0)
    X_sky = np.concatenate(all_X_sky, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = np.concatenate(all_meta, axis=0)

    # Scaling
    if mode == "train":
        scaler = StandardScaler()
        # Fit on Traj and Sky features separately or together?
        # Usually separately or concatenated. Let's concatenate for scaling simplicity or keep separate.
        # Given they go into different network heads, separate scalers or one big one?
        # Let's use one scaler for all input features concatenated.
        # Actually, simpler to manage if we just scale X_traj and X_sky independently or just flatten everything.
        # Let's scale them separately to handle the dimensions easily.

        # We will use a dictionary to hold scalers
        scaler = {"traj": StandardScaler(), "sky": StandardScaler()}
        X_traj = scaler["traj"].fit_transform(X_traj)
        X_sky = scaler["sky"].fit_transform(X_sky)

        # Save scaler
        joblib.dump(scaler, os.path.join(CACHE_DIR, "scaler.joblib"))

    else:
        if scaler is None:
            # Try loading
            scaler_path = os.path.join(CACHE_DIR, "scaler.joblib")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                raise ValueError("Scaler must be provided for val/test mode.")

        X_traj = scaler["traj"].transform(X_traj)
        X_sky = scaler["sky"].transform(X_sky)

    dataset = GNSSWindowDataset(X_traj, X_sky, y if mode != "test" else None, meta)

    return dataset, scaler
