import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from library.config import Config
from library.utils import latlon_to_meters, meters_to_latlon

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA) using WGS84 constants.
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

    # Avoid division by zero for altitude
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e**2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


class JSONStandardScaler:
    """
    Standard Scaler that saves/loads state to/from JSON to avoid pickle.
    """

    def __init__(self):
        self.mean = None
        self.scale = None
        self.feature_names = None

    def fit(self, X, feature_names=None):
        self.mean = np.nanmean(X, axis=0)
        self.scale = np.nanstd(X, axis=0)
        # Avoid division by zero
        self.scale[self.scale == 0] = 1.0
        self.feature_names = feature_names
        return self

    def transform(self, X):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean) / self.scale

    def save(self, path):
        data = {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feature_names": self.feature_names,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        self.feature_names = data.get("feature_names")
        return self


# -------------------------------------------------------------------------
# Data Processing Core
# -------------------------------------------------------------------------


def process_trip(trip_id, df_gnss, df_gt, is_train, window_size):
    """
    Process a single trip: calculate WLS LLA, derived features, and generate windows.
    """
    # Sort by time
    df_gnss = df_gnss.sort_values("utcTimeMillis").reset_index(drop=True)

    # 1. Calculate Baseline WLS LLA from ECEF
    # Note: device_gnss.csv contains WlsPosition[X/Y/Z]EcefMeters
    # We aggregate to 1 row per epoch (mean of signal features, first of position)

    # Features to aggregate
    agg_dict = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "Cn0DbHz": ["mean", "std"],
        "SvElevationDegrees": ["mean", "std"],
        "SvAzimuthDegrees": ["mean", "std"],
        "RawPseudorangeUncertaintyMeters": ["mean", "std"],
        "Svid": "count",  # Signal count
    }

    # Group by epoch
    df_epoch = df_gnss.groupby("utcTimeMillis").agg(agg_dict)

    # Flatten MultiIndex columns
    df_epoch.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_epoch.columns.values
    ]

    # Rename for clarity
    rename_map = {
        "WlsPositionXEcefMeters_first": "WlsX",
        "WlsPositionYEcefMeters_first": "WlsY",
        "WlsPositionZEcefMeters_first": "WlsZ",
        "Cn0DbHz_mean": "mean_Cn0",
        "Cn0DbHz_std": "std_Cn0",
        "SvElevationDegrees_mean": "mean_SvElevation",
        "SvElevationDegrees_std": "std_SvElevation",
        "SvAzimuthDegrees_mean": "mean_SvAzimuth",
        "SvAzimuthDegrees_std": "std_SvAzimuth",
        "RawPseudorangeUncertaintyMeters_mean": "mean_Uncertainty",
        "RawPseudorangeUncertaintyMeters_std": "std_Uncertainty",
        "Svid_count": "mean_SignalCount",  # Technically count per epoch
    }
    df_epoch = df_epoch.rename(columns=rename_map).reset_index()

    # Convert WLS ECEF to LLA
    wls_lat, wls_lon, wls_alt = ecef_to_lla(
        df_epoch["WlsX"].values, df_epoch["WlsY"].values, df_epoch["WlsZ"].values
    )
    df_epoch["WlsLat"] = wls_lat
    df_epoch["WlsLon"] = wls_lon
    df_epoch["WlsAlt"] = wls_alt

    # Fill NaNs in Sky features (e.g. std of 1 satellite is NaN)
    df_epoch = df_epoch.fillna(0)

    # 2. Merge with Ground Truth (if training) or Metadata (to filter required timestamps)
    # df_gt contains the target timestamps we care about.
    # We perform an asof merge or exact merge. Given the problem, exact merge on millis is usually expected
    # but let's be safe with a tolerance if needed. Here we assume exact alignment or subset.

    # In test mode, df_gt is actually the test_metadata which has the timestamps we need to predict.
    # In train mode, df_gt is the ground truth.

    # We keep all GNSS epochs to form windows, but we mark the ones that are targets.
    target_timestamps = set(df_gt["UnixTimeMillis"].values)

    # 3. Generate Windows
    # We need to create a window of size N centered at each epoch.
    # Sequence features: Relative Lat/Lon/Alt, Velocity, Cn0, Uncertainty

    # Pre-calculate sequence arrays
    lat = df_epoch["WlsLat"].values
    lon = df_epoch["WlsLon"].values
    alt = df_epoch["WlsAlt"].values
    cn0 = df_epoch["mean_Cn0"].values
    unc = df_epoch["mean_Uncertainty"].values
    times = df_epoch["utcTimeMillis"].values

    num_epochs = len(df_epoch)
    half_window = window_size // 2

    X_seq_list = []
    X_sky_list = []
    y_list = []
    meta_list = []

    # Lookup for GT
    if is_train:
        gt_lookup = df_gt.set_index("UnixTimeMillis")[
            ["LatitudeDegrees", "LongitudeDegrees"]
        ].to_dict("index")

    # Iterate through epochs where a full window can be formed
    for i in range(half_window, num_epochs - half_window):
        center_time = times[i]

        # Filter: Only process if this timestamp is a required target
        if center_time not in target_timestamps:
            continue

        # Indices for the window
        start_idx = i - half_window
        end_idx = i + half_window + 1  # Exclusive

        # --- Sequence Features ---
        # 1. Relative Centering
        center_lat = lat[i]
        center_lon = lon[i]
        center_alt = alt[i]

        window_lat = lat[start_idx:end_idx]
        window_lon = lon[start_idx:end_idx]
        window_alt = alt[start_idx:end_idx]

        # Convert to meters relative to center
        rel_n, rel_e = latlon_to_meters(window_lat, window_lon, center_lat, center_lon)
        rel_u = window_alt - center_alt

        # 2. Dynamics (Velocity)
        # Simple difference: pos[t] - pos[t-1]. For first element, duplicate or 0.
        # We compute diffs on the relative meters.
        vel_n = np.diff(rel_n, prepend=rel_n[0])
        vel_e = np.diff(rel_e, prepend=rel_e[0])
        vel_u = np.diff(rel_u, prepend=rel_u[0])

        # 3. Stack Sequence
        # Features: [rel_n, rel_e, rel_u, vel_n, vel_e, vel_u, cn0, unc]
        win_cn0 = cn0[start_idx:end_idx]
        win_unc = unc[start_idx:end_idx]

        seq_feat = np.stack(
            [rel_n, rel_e, rel_u, vel_n, vel_e, vel_u, win_cn0, win_unc], axis=1
        )  # Shape (Window, 8)

        # --- Sky-State Context ---
        # Aggregate over the window
        # We use the pre-aggregated epoch stats and aggregate them again over the window
        # This gives a sense of the "scene" (e.g. consistently low elevation)
        sky_cols = [
            "mean_Cn0",
            "std_Cn0",
            "mean_SvElevation",
            "std_SvElevation",
            "mean_SvAzimuth",
            "std_SvAzimuth",
            "mean_Uncertainty",
            "std_Uncertainty",
            "mean_SignalCount",
        ]
        win_sky_data = df_epoch.iloc[start_idx:end_idx][sky_cols].values
        sky_feat = np.mean(win_sky_data, axis=0)  # Shape (9,)

        # --- Target ---
        target = np.array([0.0, 0.0])  # Dummy for test
        if is_train:
            if center_time in gt_lookup:
                gt_lat = gt_lookup[center_time]["LatitudeDegrees"]
                gt_lon = gt_lookup[center_time]["LongitudeDegrees"]

                # Target is residual in meters from WLS center
                t_n, t_e = latlon_to_meters(gt_lat, gt_lon, center_lat, center_lon)
                target = np.array([t_e, t_n])  # East, North
            else:
                # Should not happen due to filter, but safety check
                continue

        X_seq_list.append(seq_feat)
        X_sky_list.append(sky_feat)
        y_list.append(target)
        meta_list.append(
            {
                "tripId": trip_id,
                "UnixTimeMillis": center_time,
                "WlsLat": center_lat,
                "WlsLon": center_lon,
            }
        )

    return X_seq_list, X_sky_list, y_list, meta_list


def load_and_process_split(metadata_path, scaler=None, fit_scaler=False, is_train=True):
    """
    Load data for a split (train/val/test), process it, and optionally fit scaler.
    """
    df_meta = pd.read_csv(metadata_path)

    # If debugging, sample trips
    if Config.DEBUG:
        unique_trips = df_meta["tripId"].unique()
        sample_trips = unique_trips[:5]  # Take first 5 trips
        df_meta = df_meta[df_meta["tripId"].isin(sample_trips)].copy()
        print(f"DEBUG: Processed subset of {len(sample_trips)} trips.")

    # Group by trip
    trips = df_meta["tripId"].unique()

    all_X_seq = []
    all_X_sky = []
    all_y = []
    all_meta = []

    print(f"Processing {len(trips)} trips...")
    for trip_id in tqdm(trips):
        trip_subset = df_meta[df_meta["tripId"] == trip_id]

        # Get file paths (assume first row has valid paths)
        gnss_rel_path = trip_subset.iloc[0]["gnss_path"]

        gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            continue

        df_gnss = pd.read_csv(gnss_path)

        # Process
        X_seq, X_sky, y, meta = process_trip(
            trip_id,
            df_gnss,
            trip_subset,  # Contains GT or Target Timestamps
            is_train,
            Config.WINDOW_SIZE,
        )

        all_X_seq.extend(X_seq)
        all_X_sky.extend(X_sky)
        all_y.extend(y)
        all_meta.extend(meta)

    # Convert to numpy
    X_seq_arr = np.array(all_X_seq, dtype=np.float32)
    X_sky_arr = np.array(all_X_sky, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)

    # Scaling
    if fit_scaler:
        scaler = JSONStandardScaler()
        # Fit on flattened sequence features + sky features separately?
        # Let's fit on them separately.
        # Sequence: (N, L, F) -> flatten to (N*L, F) for stats
        N, L, F_seq = X_seq_arr.shape
        scaler.fit_seq_mean = np.nanmean(X_seq_arr.reshape(-1, F_seq), axis=0)
        scaler.fit_seq_scale = np.nanstd(X_seq_arr.reshape(-1, F_seq), axis=0)
        scaler.fit_seq_scale[scaler.fit_seq_scale == 0] = 1.0

        scaler.fit_sky_mean = np.nanmean(X_sky_arr, axis=0)
        scaler.fit_sky_scale = np.nanstd(X_sky_arr, axis=0)
        scaler.fit_sky_scale[scaler.fit_sky_scale == 0] = 1.0

        # Save custom scaler attributes manually since our class is simple
        # We'll just patch the transform method logic or use two scalers.
        # For simplicity in this script, let's just normalize manually here and save stats.
        scaler_data = {
            "seq_mean": scaler.fit_seq_mean.tolist(),
            "seq_scale": scaler.fit_seq_scale.tolist(),
            "sky_mean": scaler.fit_sky_mean.tolist(),
            "sky_scale": scaler.fit_sky_scale.tolist(),
        }
        with open(Config.CACHE_SCALER, "w") as f:
            json.dump(scaler_data, f)

    else:
        # Load scaler
        if scaler is None and os.path.exists(Config.CACHE_SCALER):
            with open(Config.CACHE_SCALER, "r") as f:
                scaler_data = json.load(f)

            # Create a dummy object to hold stats
            class ScalerContainer:
                pass

            scaler = ScalerContainer()
            scaler.fit_seq_mean = np.array(scaler_data["seq_mean"])
            scaler.fit_seq_scale = np.array(scaler_data["seq_scale"])
            scaler.fit_sky_mean = np.array(scaler_data["sky_mean"])
            scaler.fit_sky_scale = np.array(scaler_data["sky_scale"])
        elif scaler is None:
            raise ValueError("Scaler not found and fit_scaler is False.")

    # Apply Transform
    X_seq_arr = (X_seq_arr - scaler.fit_seq_mean) / scaler.fit_seq_scale
    X_sky_arr = (X_sky_arr - scaler.fit_sky_mean) / scaler.fit_sky_scale

    # Handle NaNs (if any remain)
    X_seq_arr = np.nan_to_num(X_seq_arr)
    X_sky_arr = np.nan_to_num(X_sky_arr)

    return X_seq_arr, X_sky_arr, y_arr, pd.DataFrame(all_meta)


def load_data(load_cached_data=True):
    """
    Main entry point to load train, validation, and test data.
    Handles caching logic.
    """
    # 1. Check Cache for Train
    if load_cached_data and os.path.exists(Config.CACHE_TRAIN_X_SEQ):
        print("Loading cached training data...")
        train_X_seq = np.load(Config.CACHE_TRAIN_X_SEQ)
        train_X_sky = np.load(Config.CACHE_TRAIN_X_SKY)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        # Load scaler for reuse
        with open(Config.CACHE_SCALER, "r") as f:
            scaler_data = json.load(f)

        class ScalerContainer:
            pass

        scaler = ScalerContainer()
        scaler.fit_seq_mean = np.array(scaler_data["seq_mean"])
        scaler.fit_seq_scale = np.array(scaler_data["seq_scale"])
        scaler.fit_sky_mean = np.array(scaler_data["sky_mean"])
        scaler.fit_sky_scale = np.array(scaler_data["sky_scale"])
    else:
        print("Processing training data...")
        train_X_seq, train_X_sky, train_y, _ = load_and_process_split(
            Config.TRAIN_METADATA_PATH, fit_scaler=True, is_train=True
        )
        # Save Cache
        np.save(Config.CACHE_TRAIN_X_SEQ, train_X_seq)
        np.save(Config.CACHE_TRAIN_X_SKY, train_X_sky)
        np.save(Config.CACHE_TRAIN_Y, train_y)
        # Scaler is saved inside load_and_process_split

        # Create a dummy scaler object for next steps
        with open(Config.CACHE_SCALER, "r") as f:
            scaler_data = json.load(f)

        class ScalerContainer:
            pass

        scaler = ScalerContainer()
        scaler.fit_seq_mean = np.array(scaler_data["seq_mean"])
        scaler.fit_seq_scale = np.array(scaler_data["seq_scale"])
        scaler.fit_sky_mean = np.array(scaler_data["sky_mean"])
        scaler.fit_sky_scale = np.array(scaler_data["sky_scale"])

    # 2. Check Cache for Validation
    if load_cached_data and os.path.exists(Config.CACHE_VAL_X_SEQ):
        print("Loading cached validation data...")
        val_X_seq = np.load(Config.CACHE_VAL_X_SEQ)
        val_X_sky = np.load(Config.CACHE_VAL_X_SKY)
        val_y = np.load(Config.CACHE_VAL_Y)
        val_meta = pd.read_parquet(Config.CACHE_VAL_META)
    else:
        print("Processing validation data...")
        val_X_seq, val_X_sky, val_y, val_meta = load_and_process_split(
            Config.VAL_METADATA_PATH, scaler=scaler, fit_scaler=False, is_train=True
        )
        np.save(Config.CACHE_VAL_X_SEQ, val_X_seq)
        np.save(Config.CACHE_VAL_X_SKY, val_X_sky)
        np.save(Config.CACHE_VAL_Y, val_y)
        val_meta.to_parquet(Config.CACHE_VAL_META)

    # 3. Check Cache for Test
    if load_cached_data and os.path.exists(Config.CACHE_TEST_X_SEQ):
        print("Loading cached test data...")
        test_X_seq = np.load(Config.CACHE_TEST_X_SEQ)
        test_X_sky = np.load(Config.CACHE_TEST_X_SKY)
        test_meta = pd.read_parquet(Config.CACHE_TEST_META)
    else:
        print("Processing test data...")
        test_X_seq, test_X_sky, _, test_meta = load_and_process_split(
            Config.TEST_METADATA_PATH, scaler=scaler, fit_scaler=False, is_train=False
        )
        np.save(Config.CACHE_TEST_X_SEQ, test_X_seq)
        np.save(Config.CACHE_TEST_X_SKY, test_X_sky)
        test_meta.to_parquet(Config.CACHE_TEST_META)

    return (
        (train_X_seq, train_X_sky, train_y),
        (val_X_seq, val_X_sky, val_y, val_meta),
        (test_X_seq, test_X_sky, test_meta),
    )


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class GNSSWindowDataset(Dataset):
    """
    PyTorch Dataset for the Sky-State Transformer.
    Returns:
        seq_features: (Window, Feats)
        sky_features: (Sky_Feats,)
        target: (2,) -> Delta East, Delta North
    """

    def __init__(self, X_seq, X_sky, y=None):
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_sky = torch.FloatTensor(X_sky)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_seq[idx], self.X_sky[idx], self.y[idx]
        else:
            return self.X_seq[idx], self.X_sky[idx]
