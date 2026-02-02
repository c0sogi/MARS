import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import ecef_to_lla, degrees_to_meters_diff


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the scaler on the training data.
        X shape: (N_samples, Window_Size, N_features)
        """
        N, W, F = X.shape
        # Flatten to fit scaler on all timesteps
        self.scaler.fit(X.reshape(-1, F))

    def transform(self, X):
        """
        Apply scaling to the data.
        """
        N, W, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(N, W, F)

    def save_scaler(self, path):
        params = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(params, f)

    def load_scaler(self, path):
        with open(path, "r") as f:
            params = json.load(f)
        self.scaler.mean_ = np.array(params["mean"])
        self.scaler.scale_ = np.array(params["scale"])
        self.scaler.var_ = np.array(params["var"])


def process_trip(trip_id, drive_id, phone_name, gnss_path, target_timestamps=None):
    """
    Loads GNSS data for a trip, aggregates by epoch, interpolates,
    creates sliding windows, and computes relative features.

    Returns:
        X_trip: (N, Window, Features)
        valid_ts: (N,) timestamps corresponding to the center of windows
        wls_center: (N, 2) [Lat, Lon] of the WLS baseline at window center
    """
    gnss_full_path = os.path.join(Config.INPUT_DIR, gnss_path)
    if not os.path.exists(gnss_full_path):
        return None, None, None

    try:
        df_gnss = pd.read_csv(gnss_full_path)
    except Exception:
        return None, None, None

    # Aggregate by epoch
    agg_funcs = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "RawPseudorangeUncertaintyMeters": "mean",
        "Cn0DbHz": "mean",
        "Svid": "count",
    }
    # Filter columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_gnss.columns}

    df_epoch = df_gnss.groupby("utcTimeMillis").agg(agg_funcs).reset_index()
    df_epoch.rename(
        columns={
            "Svid": "sat_count",
            "Cn0DbHz": "cn0",
            "RawPseudorangeUncertaintyMeters": "raw_pr_unc",
        },
        inplace=True,
    )

    # Convert WLS to LLA
    lats, lons, alts = ecef_to_lla(
        df_epoch["WlsPositionXEcefMeters"].values,
        df_epoch["WlsPositionYEcefMeters"].values,
        df_epoch["WlsPositionZEcefMeters"].values,
    )
    df_epoch["wls_lat"] = lats
    df_epoch["wls_lon"] = lons
    df_epoch["wls_alt"] = alts

    # Handle missing epochs via Interpolation (1Hz grid)
    min_time = df_epoch["utcTimeMillis"].min()
    max_time = df_epoch["utcTimeMillis"].max()
    full_range = np.arange(min_time, max_time + 1000, 1000)

    df_epoch = df_epoch.set_index("utcTimeMillis").reindex(full_range)
    df_epoch = df_epoch.interpolate(method="linear").ffill().bfill().reset_index()
    df_epoch.rename(columns={"index": "utcTimeMillis"}, inplace=True)

    # Calculate Velocities (m/s approx)
    # Lat diff in meters
    df_epoch["vel_lat_m"] = df_epoch["wls_lat"].diff().fillna(0) * Config.LAT_SCALE
    # Lon diff in meters
    mean_lat = df_epoch["wls_lat"].mean()
    df_epoch["vel_lon_m"] = (
        df_epoch["wls_lon"].diff().fillna(0)
        * Config.LAT_SCALE
        * np.cos(np.radians(mean_lat))
    )
    df_epoch["vel_alt_m"] = df_epoch["wls_alt"].diff().fillna(0)

    # Prepare features for windowing
    # Order: [lat, lon, vel_lat, vel_lon, vel_alt, unc, cn0, sat]
    feature_data = df_epoch[
        [
            "wls_lat",
            "wls_lon",
            "vel_lat_m",
            "vel_lon_m",
            "vel_alt_m",
            "raw_pr_unc",
            "cn0",
            "sat_count",
        ]
    ].values

    timestamps = df_epoch["utcTimeMillis"].values

    # Pad data to allow windows at edges
    pad_size = Config.WINDOW_SIZE // 2
    feature_data_padded = np.pad(
        feature_data, ((pad_size, pad_size), (0, 0)), mode="edge"
    )

    # Create Sliding Windows
    # Shape: (N_epochs, Window_Size, Features)
    windows = np.lib.stride_tricks.sliding_window_view(
        feature_data_padded, window_shape=Config.WINDOW_SIZE, axis=0
    )

    # Filter for target timestamps
    if target_timestamps is not None:
        # Find indices of target timestamps in the interpolated timeline
        # Use searchsorted to find closest index
        indices = np.searchsorted(timestamps, target_timestamps)
        indices = np.clip(indices, 0, len(timestamps) - 1)

        # Verify match within tolerance (e.g. 1.5s)
        matched_times = timestamps[indices]
        diffs = np.abs(matched_times - target_timestamps)
        valid_mask = diffs < 1500

        final_indices = indices[valid_mask]
        valid_ts = target_timestamps[valid_mask]

        if len(final_indices) == 0:
            return None, None, None

        X_trip = windows[final_indices]
    else:
        X_trip = windows
        valid_ts = timestamps

    # Extract Center WLS positions (for target computation and reconstruction)
    c = Config.WINDOW_SIZE // 2
    center_lats = X_trip[:, c, 0:1]  # (N, 1)
    center_lons = X_trip[:, c, 1:2]  # (N, 1)

    wls_center = np.hstack([center_lats, center_lons])

    # Compute Relative Coordinates within windows
    # Copy to avoid modifying original if shared
    X_trip = X_trip.copy()

    # Rel Lat (Meters)
    d_lat = X_trip[:, :, 0] - center_lats
    X_trip[:, :, 0] = d_lat * Config.LAT_SCALE

    # Rel Lon (Meters)
    d_lon = X_trip[:, :, 1] - center_lons
    cos_term = np.cos(np.radians(center_lats))
    X_trip[:, :, 1] = d_lon * Config.LAT_SCALE * cos_term

    return X_trip, valid_ts, wls_center


def get_data(load_cached_data=True):
    """
    Main function to load and process data.
    """
    # Check cache
    if load_cached_data:
        if (
            os.path.exists(Config.CACHE_TRAIN_X)
            and os.path.exists(Config.CACHE_TRAIN_Y)
            and os.path.exists(Config.CACHE_VAL_X)
            and os.path.exists(Config.CACHE_VAL_Y)
            and os.path.exists(Config.CACHE_TEST_X)
            and os.path.exists(Config.CACHE_TEST_META)
            and os.path.exists(Config.CACHE_SCALER)
        ):

            print("Loading cached data...")
            train_X = np.load(Config.CACHE_TRAIN_X)
            train_y = np.load(Config.CACHE_TRAIN_Y)
            val_X = np.load(Config.CACHE_VAL_X)
            val_y = np.load(Config.CACHE_VAL_Y)
            test_X = np.load(Config.CACHE_TEST_X)
            test_meta = pd.read_parquet(Config.CACHE_TEST_META)

            return train_X, train_y, val_X, val_y, test_X, test_meta

    print("Processing data from scratch...")

    # Load Metadata
    train_meta_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

    feature_engineer = FeatureEngineer()

    def process_split(meta_df, is_test=False):
        X_list = []
        y_list = []
        meta_list = []

        # Group by trip to process sequentially
        for trip_id, group in meta_df.groupby("tripId"):
            first = group.iloc[0]
            drive_id = first["drive_id"]
            phone_name = first["phone_name"]
            gnss_path = first["gnss_path"]

            target_ts = group["UnixTimeMillis"].values

            X_trip, valid_ts, wls_center = process_trip(
                trip_id, drive_id, phone_name, gnss_path, target_timestamps=target_ts
            )

            if X_trip is None:
                continue

            X_list.append(X_trip)

            # Align metadata
            valid_group = group[group["UnixTimeMillis"].isin(valid_ts)].copy()
            # Ensure order matches valid_ts
            valid_group = (
                valid_group.set_index("UnixTimeMillis").reindex(valid_ts).reset_index()
            )
            meta_list.append(valid_group)

            if not is_test:
                # Compute Targets: GT - WLS (Meters)
                gt_lat = valid_group["LatitudeDegrees"].values
                gt_lon = valid_group["LongitudeDegrees"].values

                wls_lat = wls_center[:, 0]
                wls_lon = wls_center[:, 1]

                d_lat_m, d_lon_m = degrees_to_meters_diff(
                    gt_lat - wls_lat, gt_lon - wls_lon, wls_lat
                )

                targets = np.stack([d_lat_m, d_lon_m], axis=1)
                y_list.append(targets)

        if not X_list:
            return None, None, None

        X_all = np.concatenate(X_list, axis=0)
        meta_all = pd.concat(meta_list, ignore_index=True)
        y_all = np.concatenate(y_list, axis=0) if not is_test else None

        return X_all, y_all, meta_all

    print("Processing Train...")
    train_X, train_y, _ = process_split(train_meta_df, is_test=False)
    print("Processing Val...")
    val_X, val_y, _ = process_split(val_meta_df, is_test=False)
    print("Processing Test...")
    test_X, _, test_meta = process_split(test_meta_df, is_test=True)

    # Scaling
    print("Fitting Scaler...")
    feature_engineer.fit(train_X)

    print("Transforming Data...")
    train_X = feature_engineer.transform(train_X)
    val_X = feature_engineer.transform(val_X)
    test_X = feature_engineer.transform(test_X)

    # Save Cache
    print("Saving Cache...")
    np.save(Config.CACHE_TRAIN_X, train_X)
    np.save(Config.CACHE_TRAIN_Y, train_y)
    np.save(Config.CACHE_VAL_X, val_X)
    np.save(Config.CACHE_VAL_Y, val_y)
    np.save(Config.CACHE_TEST_X, test_X)
    test_meta.to_parquet(Config.CACHE_TEST_META)
    feature_engineer.save_scaler(Config.CACHE_SCALER)

    return train_X, train_y, val_X, val_y, test_X, test_meta
