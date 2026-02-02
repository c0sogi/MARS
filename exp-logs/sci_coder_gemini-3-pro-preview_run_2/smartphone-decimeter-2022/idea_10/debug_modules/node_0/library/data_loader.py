import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

# Import configuration and utility functions
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALER_PATH,
    WINDOW_SIZE,
    LAT_DEG_TO_METERS,
    TRAJECTORY_FEATURES,
    SKY_FEATURES,
    TARGET_FEATURES,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import ecef_to_llh, degrees_to_meters_diff, haversine_distance

# Set random seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


class GNSSDataset(Dataset):
    """
    PyTorch Dataset for GNSS data.
    """

    def __init__(self, trajectory_data, sky_data, targets=None, metadata=None):
        self.trajectory_data = torch.FloatTensor(trajectory_data)
        self.sky_data = torch.FloatTensor(sky_data)
        self.metadata = metadata

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, idx):
        # Trajectory: (Channels, Length) -> PyTorch Conv1D expects (Batch, Channels, Length)
        # Our data is (N, Length, Channels), so we transpose to (Channels, Length)
        traj = self.trajectory_data[idx].transpose(0, 1)
        sky = self.sky_data[idx]

        if self.targets is not None:
            target = self.targets[idx]
            return traj, sky, target
        else:
            return traj, sky


class CustomScaler:
    """
    Standard Scaler for 3D (Trajectory) and 2D (Sky) data.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, traj_data, sky_data):
        # traj_data shape: (N, Window, Channels)
        # Flatten window dimension for stats calculation
        N, W, C_traj = traj_data.shape
        traj_flat = traj_data.reshape(-1, C_traj)

        self.means["traj"] = np.nanmean(traj_flat, axis=0).tolist()
        self.stds["traj"] = np.nanstd(traj_flat, axis=0).tolist()

        # sky_data shape: (N, Channels)
        self.means["sky"] = np.nanmean(sky_data, axis=0).tolist()
        self.stds["sky"] = np.nanstd(sky_data, axis=0).tolist()

        # Avoid division by zero
        self.stds["traj"] = [s if s > 1e-6 else 1.0 for s in self.stds["traj"]]
        self.stds["sky"] = [s if s > 1e-6 else 1.0 for s in self.stds["sky"]]

    def transform(self, traj_data, sky_data):
        traj_norm = (traj_data - np.array(self.means["traj"])) / np.array(
            self.stds["traj"]
        )
        sky_norm = (sky_data - np.array(self.means["sky"])) / np.array(self.stds["sky"])

        # Handle potential NaNs created during scaling or existing before
        traj_norm = np.nan_to_num(traj_norm)
        sky_norm = np.nan_to_num(sky_norm)

        return traj_norm, sky_norm

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"means": self.means, "stds": self.stds}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.means = data["means"]
            self.stds = data["stds"]


def load_gnss_data(gnss_path):
    """
    Load and aggregate GNSS data by epoch.
    """
    full_path = os.path.join(INPUT_DIR, gnss_path)
    if not os.path.exists(full_path):
        return None

    df = pd.read_csv(full_path)

    # Rename time column for consistency
    if "utcTimeMillis" in df.columns:
        df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # Aggregate features per epoch
    # We need WLS position (baseline) and signal stats
    agg_funcs = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "Cn0DbHz": ["mean", "std"],
        "SvElevationDegrees": ["mean", "std"],
        "SvAzimuthDegrees": ["mean", "std"],
        "RawPseudorangeUncertaintyMeters": "mean",
        "Svid": "count",  # Satellite count
    }

    # Filter for existing columns
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

    df_agg = df.groupby("UnixTimeMillis").agg(agg_funcs)

    # Flatten multi-level columns
    df_agg.columns = ["_".join(col).strip("_") for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)

    # Rename to match config SKY_FEATURES and internal logic
    rename_map = {
        "Cn0DbHz_mean": "mean_cn0_sky",
        "Cn0DbHz_std": "std_cn0",  # Not in config SKY_FEATURES but useful intermediate
        "SvElevationDegrees_mean": "mean_elev",
        "SvElevationDegrees_std": "std_elev",
        "SvAzimuthDegrees_mean": "mean_azim",
        "SvAzimuthDegrees_std": "std_azim",
        "Svid_count": "sat_count",
        "RawPseudorangeUncertaintyMeters_mean": "raw_uncertainty",  # Used in trajectory
        "Cn0DbHz_mean": "raw_cn0",  # Re-using mean CN0 for trajectory feature 'raw_cn0' roughly
    }

    # Handle the fact that we need 'raw_cn0' for trajectory (per epoch) and 'mean_cn0_sky' for sky
    # In this aggregation, they are the same value (mean over satellites at that epoch)
    # The config implies trajectory features are per epoch.

    df_agg.rename(columns=rename_map, inplace=True)

    # Fill NaNs in std columns (single satellite case)
    for col in df_agg.columns:
        if "std" in col:
            df_agg[col] = df_agg[col].fillna(0)

    # Convert WLS ECEF to LLA
    if "WlsPositionXEcefMeters_first" in df_agg.columns:
        lat, lon, alt = ecef_to_llh(
            df_agg["WlsPositionXEcefMeters_first"].values,
            df_agg["WlsPositionYEcefMeters_first"].values,
            df_agg["WlsPositionZEcefMeters_first"].values,
        )
        df_agg["wls_lat"] = lat
        df_agg["wls_lon"] = lon
        df_agg["wls_alt"] = alt

    return df_agg


def process_trip(trip_metadata, is_train=True):
    """
    Process a single trip: align GNSS/GT, create windows, compute features.
    """
    trip_id = trip_metadata["tripId"]
    gnss_path = trip_metadata["gnss_path"]

    # Load GNSS Data
    df_gnss = load_gnss_data(gnss_path)
    if df_gnss is None or df_gnss.empty:
        return None, None, None, None

    # Load Ground Truth or Test Timestamps
    if is_train:
        # For training, we use the metadata itself which contains GT
        # Filter metadata for this trip
        # Ideally, we passed a row or a dataframe subset.
        # Assuming trip_metadata is a Series from the metadata dataframe,
        # we might need to load the full GT for alignment if timestamps differ.
        # However, the metadata CSV *is* the GT for train.
        # But `process_trip` is called in a loop over unique trips.
        # Let's assume we pass the subset of the metadata dataframe corresponding to this trip.
        df_gt = trip_metadata.sort_values("UnixTimeMillis")
    else:
        # For test, trip_metadata is the subset of test_metadata.csv for this trip
        df_gt = trip_metadata.sort_values("UnixTimeMillis")

    # Merge GNSS and GT/Test Targets
    # We use an exact merge on UnixTimeMillis as per EDA findings
    df_merged = pd.merge(df_gt, df_gnss, on="UnixTimeMillis", how="inner")

    if df_merged.empty:
        return None, None, None, None

    # Sort by time
    df_merged = df_merged.sort_values("UnixTimeMillis").reset_index(drop=True)

    # Prepare Arrays
    timestamps = df_merged["UnixTimeMillis"].values
    wls_lat = df_merged["wls_lat"].values
    wls_lon = df_merged["wls_lon"].values
    wls_alt = df_merged["wls_alt"].values

    # Features for Trajectory
    # We need to construct the sequence.
    # Since we need a window centered at t, we need data from t - W/2 to t + W/2.
    # We will iterate through the merged dataframe. If the time continuity is broken, we skip.

    traj_list = []
    sky_list = []
    target_list = []
    meta_list = []

    half_window = WINDOW_SIZE // 2

    # Pre-calculate velocity (simple difference)
    # Pad with zeros or edge values
    vel_lat = np.gradient(wls_lat) * LAT_DEG_TO_METERS
    vel_lon = np.gradient(wls_lon) * LAT_DEG_TO_METERS * np.cos(np.radians(wls_lat))
    vel_alt = np.gradient(wls_alt)

    # Ensure required columns exist
    req_traj_cols = ["raw_cn0", "raw_uncertainty"]
    req_sky_cols = [
        "mean_elev",
        "std_elev",
        "mean_azim",
        "std_azim",
        "mean_cn0_sky",
        "sat_count",
    ]  # 'mean_cn0_sky' is 'raw_cn0' in df_agg

    # Map config feature names to dataframe columns
    # Note: in load_gnss_data we renamed 'Cn0DbHz_mean' to 'raw_cn0' which is used for both
    # 'raw_cn0' (trajectory) and 'mean_cn0_sky' (sky context).
    # We need to duplicate the column for clarity or just access it.
    df_merged["mean_cn0_sky"] = df_merged["raw_cn0"]

    for col in req_traj_cols + req_sky_cols:
        if col not in df_merged.columns:
            df_merged[col] = 0.0  # Fill missing with 0

    # Convert to numpy for speed
    data_matrix = df_merged.to_dict("list")

    n_samples = len(df_merged)

    for i in range(n_samples):
        # Check window bounds
        start_idx = i - half_window
        end_idx = i + half_window + 1  # +1 because slice is exclusive

        if start_idx < 0 or end_idx > n_samples:
            continue

        # Check time continuity (approximate)
        # We expect 1000ms diffs. Allow some tolerance (e.g., < 2000ms gaps)
        t_window = timestamps[start_idx:end_idx]
        t_diffs = np.diff(t_window)
        if np.any(t_diffs > 2000):  # If any gap is larger than 2s, skip
            continue

        # Center epoch data
        center_lat = wls_lat[i]
        center_lon = wls_lon[i]
        center_alt = wls_alt[i]

        # Trajectory Features Construction
        # 1. Relative Lat/Lon/Alt in meters
        win_lat = wls_lat[start_idx:end_idx]
        win_lon = wls_lon[start_idx:end_idx]
        win_alt = wls_alt[start_idx:end_idx]

        d_lat_m, d_lon_m = degrees_to_meters_diff(
            win_lat - center_lat, win_lon - center_lon, center_lat
        )
        d_alt_m = win_alt - center_alt

        # 2. Velocities
        win_vel_lat = vel_lat[start_idx:end_idx]
        win_vel_lon = vel_lon[start_idx:end_idx]
        win_vel_alt = vel_alt[start_idx:end_idx]

        # 3. Signal Metrics
        win_cn0 = np.array(data_matrix["raw_cn0"][start_idx:end_idx])
        win_unc = np.array(data_matrix["raw_uncertainty"][start_idx:end_idx])

        # Stack Trajectory Features: (Window, Features)
        # Order must match TRAJECTORY_FEATURES in config
        # ['rel_lat_m', 'rel_lon_m', 'rel_alt_m', 'vel_lat_m', 'vel_lon_m', 'vel_alt_m', 'raw_cn0', 'raw_uncertainty']
        traj_sample = np.stack(
            [
                d_lat_m,
                d_lon_m,
                d_alt_m,
                win_vel_lat,
                win_vel_lon,
                win_vel_alt,
                win_cn0,
                win_unc,
            ],
            axis=1,
        )

        # Sky Context Features (Aggregated over window)
        # We take the mean of the pre-aggregated epoch stats over the window
        # This gives a smoother "sky view"
        sky_feats_vals = []
        for feat in SKY_FEATURES:
            vals = np.array(data_matrix[feat][start_idx:end_idx])
            sky_feats_vals.append(np.mean(vals))

        sky_sample = np.array(sky_feats_vals)

        # Target (Residuals)
        if is_train:
            gt_lat = data_matrix["LatitudeDegrees"][i]
            gt_lon = data_matrix["LongitudeDegrees"][i]

            t_lat_m, t_lon_m = degrees_to_meters_diff(
                gt_lat - center_lat, gt_lon - center_lon, center_lat
            )
            target_sample = np.array([t_lat_m, t_lon_m])
        else:
            target_sample = np.array([0.0, 0.0])  # Dummy

        # Metadata
        meta_sample = {
            "tripId": trip_id,
            "UnixTimeMillis": timestamps[i],
            "wls_lat": center_lat,
            "wls_lon": center_lon,
        }

        traj_list.append(traj_sample)
        sky_list.append(sky_sample)
        target_list.append(target_sample)
        meta_list.append(meta_sample)

    return traj_list, sky_list, target_list, meta_list


def process_dataset(metadata_path, cache_path, load_cached_data=True, is_train=True):
    """
    Main processing function to generate dataset arrays.
    """
    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            # We store as a directory of npy files or a dict in npz?
            # Let's use npz for simplicity given the constraints.
            # Wait, requirements said "Directory Safety: Ensure .../idea_10/ exists".
            # And "Use parquet or npy".
            # Let's save arrays as .npy and meta as .parquet

            base_name = os.path.splitext(os.path.basename(cache_path))[0]
            traj_path = os.path.join(WORKING_DIR, f"{base_name}_traj.npy")
            sky_path = os.path.join(WORKING_DIR, f"{base_name}_sky.npy")
            target_path = os.path.join(WORKING_DIR, f"{base_name}_target.npy")
            meta_path = os.path.join(WORKING_DIR, f"{base_name}_meta.parquet")

            if os.path.exists(traj_path) and os.path.exists(meta_path):
                X_traj = np.load(traj_path)
                X_sky = np.load(sky_path)
                y = np.load(target_path)
                meta = pd.read_parquet(meta_path)
                return X_traj, X_sky, y, meta
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing dataset from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    # Debugging limit
    if DEBUG_SAMPLE_SIZE:
        df_meta = df_meta.iloc[:DEBUG_SAMPLE_SIZE]

    unique_trips = df_meta["tripId"].unique()

    all_traj = []
    all_sky = []
    all_targets = []
    all_meta = []

    for trip_id in tqdm(unique_trips, desc="Processing Trips"):
        trip_subset = df_meta[df_meta["tripId"] == trip_id]
        # We pass the first row to get paths, but we need the whole subset for GT alignment
        # Actually process_trip re-merges based on paths in the first row.
        # So we pass the subset.

        # Ensure paths are correct. The metadata file has paths.
        # We take the first row to extract paths.
        # But wait, df_meta contains GT for train.

        t_list, s_list, y_list, m_list = process_trip(trip_subset, is_train=is_train)

        if t_list:
            all_traj.extend(t_list)
            all_sky.extend(s_list)
            all_targets.extend(y_list)
            all_meta.extend(m_list)

    if not all_traj:
        raise ValueError("No data processed! Check paths and logic.")

    X_traj = np.array(all_traj, dtype=np.float32)
    X_sky = np.array(all_sky, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    meta = pd.DataFrame(all_meta)

    # 3. Save to cache
    base_name = os.path.splitext(os.path.basename(cache_path))[0]
    np.save(os.path.join(WORKING_DIR, f"{base_name}_traj.npy"), X_traj)
    np.save(os.path.join(WORKING_DIR, f"{base_name}_sky.npy"), X_sky)
    np.save(os.path.join(WORKING_DIR, f"{base_name}_target.npy"), y)
    meta.to_parquet(os.path.join(WORKING_DIR, f"{base_name}_meta.parquet"))

    return X_traj, X_sky, y, meta


def get_train_val_loaders(load_cached_data=True):
    """
    Generate DataLoaders for training and validation.
    """
    # Process Train
    print("Preparing Train Data...")
    train_traj, train_sky, train_y, _ = process_dataset(
        TRAIN_METADATA_PATH, TRAIN_CACHE_PATH, load_cached_data, is_train=True
    )

    # Process Val
    print("Preparing Validation Data...")
    val_traj, val_sky, val_y, _ = process_dataset(
        VAL_METADATA_PATH, VAL_CACHE_PATH, load_cached_data, is_train=True
    )

    # Scale Data
    scaler = CustomScaler()
    if load_cached_data and os.path.exists(SCALER_PATH):
        print("Loading scaler...")
        scaler.load(SCALER_PATH)
    else:
        print("Fitting scaler on training data...")
        scaler.fit(train_traj, train_sky)
        scaler.save(SCALER_PATH)

    train_traj, train_sky = scaler.transform(train_traj, train_sky)
    val_traj, val_sky = scaler.transform(val_traj, val_sky)

    # Create Datasets
    train_ds = GNSSDataset(train_traj, train_sky, train_y)
    val_ds = GNSSDataset(val_traj, val_sky, val_y)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Generate DataLoader for testing.
    """
    print("Preparing Test Data...")
    test_traj, test_sky, test_y, test_meta = process_dataset(
        TEST_METADATA_PATH, TEST_CACHE_PATH, load_cached_data, is_train=False
    )

    # Load Scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found! Run training first.")

    scaler = CustomScaler()
    scaler.load(SCALER_PATH)

    test_traj, test_sky = scaler.transform(test_traj, test_sky)

    test_ds = GNSSDataset(test_traj, test_sky, targets=None, metadata=test_meta)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_meta
