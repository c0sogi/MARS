import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import ecef_to_lla, latlon_to_meters_diff


class GNSSTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for GNSS Trajectory sequences.
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


def aggregate_imu_features(imu_df, gnss_timestamps):
    """
    Aggregates IMU data aligned to GNSS timestamps.
    Uses rounding to nearest second for efficient grouping.
    """
    if imu_df.empty:
        # Return empty dataframe with expected columns if no IMU data
        columns = Config.IMU_FEATURES
        return pd.DataFrame(
            np.zeros((len(gnss_timestamps), len(columns))),
            columns=columns,
            index=gnss_timestamps,
        )

    # Calculate magnitudes
    imu_df["AccelMag"] = np.sqrt(
        imu_df["MeasurementX"] ** 2
        + imu_df["MeasurementY"] ** 2
        + imu_df["MeasurementZ"] ** 2
    )
    # Gyro columns might not exist in all datasets, handle gracefully
    if (
        "MeasurementX_gyro" in imu_df.columns
    ):  # Assuming gyro columns might be named differently or exist
        # Based on EDA, only MeasurementX/Y/Z exist in device_imu.csv which mixes accel/gyro/mag
        # The MessageType column distinguishes them.
        pass

    # Filter and pivot based on MessageType
    # MessageType: UncalAccel, UncalGyro, UncalMag
    accel_df = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
    gyro_df = imu_df[imu_df["MessageType"] == "UncalGyro"].copy()

    # Calculate magnitudes
    accel_df["Mag"] = np.sqrt(
        accel_df["MeasurementX"] ** 2
        + accel_df["MeasurementY"] ** 2
        + accel_df["MeasurementZ"] ** 2
    )
    gyro_df["Mag"] = np.sqrt(
        gyro_df["MeasurementX"] ** 2
        + gyro_df["MeasurementY"] ** 2
        + gyro_df["MeasurementZ"] ** 2
    )

    # Binning by rounding to nearest second (1000ms)
    accel_df["time_bin"] = np.round(accel_df["utcTimeMillis"] / 1000).astype(int) * 1000
    gyro_df["time_bin"] = np.round(gyro_df["utcTimeMillis"] / 1000).astype(int) * 1000

    # Aggregate
    accel_agg = (
        accel_df.groupby("time_bin")["Mag"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "AccelMag_Mean", "std": "AccelMag_Std"})
    )
    gyro_agg = (
        gyro_df.groupby("time_bin")["Mag"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "GyroMag_Mean", "std": "GyroMag_Std"})
    )

    # Merge
    imu_agg = pd.merge(
        accel_agg, gyro_agg, left_index=True, right_index=True, how="outer"
    )

    # Reindex to GNSS timestamps (also binned)
    gnss_bins = np.round(gnss_timestamps / 1000).astype(int) * 1000

    # Map aggregated values to GNSS epochs
    # We use reindex to align with the GNSS timeline, filling missing with 0
    imu_features = imu_agg.reindex(gnss_bins).fillna(0).reset_index(drop=True)

    return imu_features


def process_trip(trip_id, gnss_path, imu_path, gt_df=None):
    """
    Loads and processes a single trip's data into a flat time-series DataFrame.
    """
    # Load Raw Data
    try:
        gnss_df = pd.read_csv(os.path.join(Config.INPUT_DIR, gnss_path))
    except FileNotFoundError:
        print(f"Warning: GNSS file not found {gnss_path}")
        return None

    imu_df = pd.DataFrame()
    if os.path.exists(os.path.join(Config.INPUT_DIR, imu_path)):
        imu_df = pd.read_csv(os.path.join(Config.INPUT_DIR, imu_path))

    # 1. Aggregate GNSS by Epoch
    # We take the first WLS position as the baseline for the epoch
    # We compute mean signal stats
    agg_funcs = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "Cn0DbHz": "mean",
        "ReceivedSvTimeUncertaintyNanos": "mean",
        "SvElevationDegrees": "mean",
        "SvAzimuthDegrees": "std",  # Spread
        "Svid": "count",
    }
    # Only aggregate columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in gnss_df.columns}

    gnss_epoch_df = gnss_df.groupby("utcTimeMillis").agg(agg_funcs).reset_index()

    # Rename columns to match Config features
    rename_map = {
        "Cn0DbHz": "MeanCn0",
        "ReceivedSvTimeUncertaintyNanos": "MeanUncertainty",
        "SvElevationDegrees": "MeanElevation",
        "SvAzimuthDegrees": "AzimuthSpread",
        "Svid": "SatelliteCount",
    }
    gnss_epoch_df.rename(columns=rename_map, inplace=True)

    # Fill missing GNSS features (e.g. if AzimuthSpread is NaN due to single satellite)
    gnss_epoch_df.fillna(0, inplace=True)

    # 2. Convert WLS ECEF to LLA
    if "WlsPositionXEcefMeters" in gnss_epoch_df.columns:
        lat, lon, alt = ecef_to_lla(
            gnss_epoch_df["WlsPositionXEcefMeters"].values,
            gnss_epoch_df["WlsPositionYEcefMeters"].values,
            gnss_epoch_df["WlsPositionZEcefMeters"].values,
        )
        gnss_epoch_df["WlsLat"] = lat
        gnss_epoch_df["WlsLon"] = lon
        gnss_epoch_df["WlsAlt"] = alt
    else:
        # Fallback if WLS missing (rare)
        gnss_epoch_df["WlsLat"] = 0.0
        gnss_epoch_df["WlsLon"] = 0.0
        gnss_epoch_df["WlsAlt"] = 0.0

    # 3. Aggregate IMU Features
    imu_feats = aggregate_imu_features(imu_df, gnss_epoch_df["utcTimeMillis"].values)

    # Concatenate GNSS and IMU
    # imu_feats is already aligned by index to gnss_epoch_df
    processed_df = pd.concat([gnss_epoch_df, imu_feats], axis=1)

    # 4. Merge Ground Truth (if available)
    if gt_df is not None:
        # Filter GT for this trip
        trip_gt = gt_df[gt_df["tripId"] == trip_id].copy()
        # Merge on time (Inner join to keep only labeled data for training)
        processed_df = pd.merge(
            processed_df,
            trip_gt[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            left_on="utcTimeMillis",
            right_on="UnixTimeMillis",
            how="inner",
        )

        # Calculate Targets (Residuals)
        d_east, d_north = latlon_to_meters_diff(
            processed_df["WlsLat"].values,
            processed_df["WlsLon"].values,
            processed_df["LatitudeDegrees"].values,
            processed_df["LongitudeDegrees"].values,
        )
        processed_df["DeltaEastMeters"] = d_east
        processed_df["DeltaNorthMeters"] = d_north

    # 5. Compute Velocity (Dynamics)
    # We compute simple finite differences
    processed_df["VelLatMeters"] = np.gradient(processed_df["WlsLat"].values) * 111320.0
    processed_df["VelLonMeters"] = (
        np.gradient(processed_df["WlsLon"].values)
        * 111320.0
        * np.cos(np.radians(processed_df["WlsLat"].values))
    )
    processed_df["VelAltMeters"] = np.gradient(processed_df["WlsAlt"].values)

    processed_df["tripId"] = trip_id

    return processed_df


def create_sliding_windows(
    df, window_size, feature_cols, target_cols=None, is_test=False
):
    """
    Converts a time-series DataFrame into sliding window arrays.
    Performs window-relative coordinate transformation.
    """
    X_list = []
    y_list = []

    half_window = window_size // 2
    n_samples = len(df)

    # Convert DataFrame to numpy for speed
    # We need specific columns
    # WLS coords for relative calculation
    wls_lat = df["WlsLat"].values
    wls_lon = df["WlsLon"].values
    wls_alt = df["WlsAlt"].values

    # Other features
    # Note: COORD_FEATURES in Config are placeholders for what we generate here
    # We take the static features (GNSS + IMU + Velocity) from the DF
    static_feat_names = [
        f
        for f in feature_cols
        if f not in ["RelLatMeters", "RelLonMeters", "RelAltMeters"]
    ]
    static_feats = df[static_feat_names].values

    if target_cols:
        targets = df[target_cols].values

    # Iterate
    # For test, we might need to be careful about indices, but usually we predict for valid WLS epochs.
    # We pad the sequence or start from index where full window is available.
    # Here we simply drop edges for simplicity, assuming long enough trips.

    for i in range(half_window, n_samples - half_window):
        # Window indices
        indices = range(i - half_window, i + half_window + 1)

        # Center coordinates
        center_lat = wls_lat[i]
        center_lon = wls_lon[i]
        center_alt = wls_alt[i]

        # 1. Calculate Relative Coordinates for the window
        # Lat/Lon diff in meters
        d_east, d_north = latlon_to_meters_diff(
            center_lat, center_lon, wls_lat[indices], wls_lon[indices]
        )
        d_alt = wls_alt[indices] - center_alt

        # 2. Get Static Features for the window
        win_static = static_feats[indices]

        # 3. Combine
        # Shape: (Window_Size, n_features)
        # We assume COORD_FEATURES order: RelLat, RelLon, RelAlt, ...
        # We stack: [RelLat, RelLon, RelAlt] + [StaticFeats]
        # Note: We must match Config.FEATURE_NAMES order.
        # Config: GNSS + IMU + COORD(RelLat, RelLon, RelAlt, VelLat, VelLon, VelAlt)
        # static_feats contains GNSS + IMU + Vel

        # Let's organize:
        # static_feats structure: [GNSS..., IMU..., Vel...]
        # We need to construct the vector per step.

        # Re-assemble based on Config.FEATURE_NAMES
        # This is slightly inefficient inside loop, but safe.
        # Better: Pre-organize static feats to match Config order excluding Rel coords.

        # Let's construct the window feature matrix
        # Columns: GNSS... IMU... RelLat RelLon RelAlt Vel...

        # Extract velocity from static (assuming they are at end or we can find them)
        # Actually, let's just build the array manually

        # Create a temporary array for this window
        window_data = np.zeros((window_size, len(Config.FEATURE_NAMES)))

        # Fill GNSS + IMU
        gnss_imu_cols = Config.GNSS_FEATURES + Config.IMU_FEATURES
        # Find indices in df
        col_indices = [df.columns.get_loc(c) for c in gnss_imu_cols]
        window_data[:, : len(gnss_imu_cols)] = df.iloc[indices, col_indices].values

        # Fill Relative Coords
        # Config order: RelLat, RelLon, RelAlt
        # Note: latlon_to_meters_diff returns d_east (Lon), d_north (Lat)
        # We map d_north -> RelLat, d_east -> RelLon
        rel_idx = len(gnss_imu_cols)
        window_data[:, rel_idx] = d_north
        window_data[:, rel_idx + 1] = d_east
        window_data[:, rel_idx + 2] = d_alt

        # Fill Velocity
        vel_cols = ["VelLatMeters", "VelLonMeters", "VelAltMeters"]
        vel_indices = [df.columns.get_loc(c) for c in vel_cols]
        window_data[:, rel_idx + 3 :] = df.iloc[indices, vel_indices].values

        X_list.append(window_data)

        if target_cols:
            y_list.append(targets[i])

    return np.array(X_list), np.array(y_list) if target_cols else None


def load_data(mode="train", load_cached_data=True, sample_fraction=None):
    """
    Main function to load and process data.

    Args:
        mode: 'train' or 'test'.
        load_cached_data: If True, tries to load .npy files from cache.
        sample_fraction: Float (0.0-1.0) to sample a fraction of trips for debugging.

    Returns:
        If train: (train_dataset, val_dataset)
        If test: (test_dataset, test_meta_df)
    """
    cache_dir = Config.WORKING_DIR

    if mode == "train":
        X_train_path = os.path.join(cache_dir, "train_X.npy")
        y_train_path = os.path.join(cache_dir, "train_y.npy")
        X_val_path = os.path.join(cache_dir, "val_X.npy")
        y_val_path = os.path.join(cache_dir, "val_y.npy")

        if (
            load_cached_data
            and os.path.exists(X_train_path)
            and os.path.exists(y_train_path)
        ):
            print("Loading cached training data...")
            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)
            X_val = np.load(X_val_path)
            y_val = np.load(y_val_path)
        else:
            print("Processing training data from scratch...")
            # Load metadata
            train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
            val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

            # Debug sampling
            if sample_fraction:
                train_trips = train_meta["tripId"].unique()
                sample_n = int(len(train_trips) * sample_fraction)
                if sample_n > 0:
                    sampled_trips = np.random.choice(
                        train_trips, sample_n, replace=False
                    )
                    train_meta = train_meta[train_meta["tripId"].isin(sampled_trips)]

            # Process trips
            def process_split(meta_df):
                X_all, y_all = [], []
                unique_trips = meta_df["tripId"].unique()
                for trip in tqdm(unique_trips, desc="Processing Trips"):
                    trip_info = meta_df[meta_df["tripId"] == trip].iloc[0]
                    # Pass the whole GT df for the trip to process_trip
                    df_trip = process_trip(
                        trip, trip_info["gnss_path"], trip_info["imu_path"], meta_df
                    )

                    if df_trip is not None and len(df_trip) > Config.WINDOW_SIZE:
                        X_trip, y_trip = create_sliding_windows(
                            df_trip,
                            Config.WINDOW_SIZE,
                            Config.FEATURE_NAMES,
                            Config.TARGET_COLUMNS,
                        )
                        X_all.append(X_trip)
                        y_all.append(y_trip)

                if X_all:
                    return np.concatenate(X_all), np.concatenate(y_all)
                return np.array([]), np.array([])

            print("Processing Train Split...")
            X_train, y_train = process_split(train_meta)
            print("Processing Val Split...")
            X_val, y_val = process_split(val_meta)

            # Save to cache
            np.save(X_train_path, X_train)
            np.save(y_train_path, y_train)
            np.save(X_val_path, X_val)
            np.save(y_val_path, y_val)

            # Fit and Save Scaler
            print("Fitting Scaler...")
            # Reshape to (N*W, F) for scaling
            N, W, F = X_train.shape
            scaler = StandardScaler()
            scaler.fit(X_train.reshape(-1, F))

            # Save scaler params manually to JSON
            scaler_params = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
            with open(Config.SCALER_PATH, "w") as f:
                json.dump(scaler_params, f)

        # Apply Scaling
        # Load scaler if not just fitted
        if not "scaler" in locals():
            with open(Config.SCALER_PATH, "r") as f:
                params = json.load(f)
            scaler = StandardScaler()
            scaler.mean_ = np.array(params["mean"])
            scaler.scale_ = np.array(params["scale"])

        # Transform
        N, W, F = X_train.shape
        X_train = scaler.transform(X_train.reshape(-1, F)).reshape(N, W, F)

        N_val, W_val, F_val = X_val.shape
        X_val = scaler.transform(X_val.reshape(-1, F_val)).reshape(N_val, W_val, F_val)

        return GNSSTrajectoryDataset(X_train, y_train), GNSSTrajectoryDataset(
            X_val, y_val
        )

    elif mode == "test":
        X_test_path = os.path.join(cache_dir, "test_X.npy")
        test_meta_path = os.path.join(cache_dir, "test_meta_processed.parquet")

        if (
            load_cached_data
            and os.path.exists(X_test_path)
            and os.path.exists(test_meta_path)
        ):
            print("Loading cached test data...")
            X_test = np.load(X_test_path)
            test_df_final = pd.read_parquet(test_meta_path)
        else:
            print("Processing test data from scratch...")
            test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

            X_list = []
            meta_list = []

            unique_trips = test_meta["tripId"].unique()
            for trip in tqdm(unique_trips, desc="Processing Test Trips"):
                trip_info = test_meta[test_meta["tripId"] == trip].iloc[0]
                # No GT for test
                df_trip = process_trip(
                    trip, trip_info["gnss_path"], trip_info["imu_path"], None
                )

                if df_trip is not None:
                    # For test, we need to extract windows centered at specific timestamps
                    # Get required timestamps for this trip
                    req_timestamps = test_meta[test_meta["tripId"] == trip][
                        "UnixTimeMillis"
                    ].values

                    # Find indices in df_trip matching timestamps
                    # We assume 1-to-1 match or close match.
                    # Using searchsorted to find closest index
                    times = df_trip["utcTimeMillis"].values
                    indices = np.searchsorted(times, req_timestamps)

                    # Clip indices to valid range
                    indices = np.clip(indices, 0, len(times) - 1)

                    # Check if timestamps match reasonably well (e.g. within 1 sec)
                    # If not, it means missing data, we might just take the closest

                    # Generate windows for these indices
                    # Re-use create_sliding_windows logic but for specific indices
                    # We can pass a dummy DF with just the rows needed? No, we need context.
                    # We implement a targeted window extractor here.

                    half_window = Config.WINDOW_SIZE // 2

                    # Pre-extract columns
                    wls_lat = df_trip["WlsLat"].values
                    wls_lon = df_trip["WlsLon"].values
                    wls_alt = df_trip["WlsAlt"].values

                    static_feat_names = [
                        f
                        for f in Config.FEATURE_NAMES
                        if f not in ["RelLatMeters", "RelLonMeters", "RelAltMeters"]
                    ]
                    static_feats = df_trip[static_feat_names].values

                    trip_X = []
                    valid_indices_mask = []

                    for k, idx in enumerate(indices):
                        # Define window bounds
                        start = idx - half_window
                        end = idx + half_window + 1

                        # Handle edge cases by padding
                        if start < 0 or end > len(df_trip):
                            # Simple edge handling: repeat border or skip
                            # Given competition nature, we must predict.
                            # We pad with the nearest valid data
                            pad_before = max(0, -start)
                            pad_after = max(0, end - len(df_trip))

                            valid_start = max(0, start)
                            valid_end = min(len(df_trip), end)

                            win_static = static_feats[valid_start:valid_end]
                            win_lat = wls_lat[valid_start:valid_end]
                            win_lon = wls_lon[valid_start:valid_end]
                            win_alt = wls_alt[valid_start:valid_end]

                            if pad_before > 0:
                                win_static = np.pad(
                                    win_static, ((pad_before, 0), (0, 0)), mode="edge"
                                )
                                win_lat = np.pad(win_lat, (pad_before, 0), mode="edge")
                                win_lon = np.pad(win_lon, (pad_before, 0), mode="edge")
                                win_alt = np.pad(win_alt, (pad_before, 0), mode="edge")
                            if pad_after > 0:
                                win_static = np.pad(
                                    win_static, ((0, pad_after), (0, 0)), mode="edge"
                                )
                                win_lat = np.pad(win_lat, (0, pad_after), mode="edge")
                                win_lon = np.pad(win_lon, (0, pad_after), mode="edge")
                                win_alt = np.pad(win_alt, (0, pad_after), mode="edge")
                        else:
                            win_static = static_feats[start:end]
                            win_lat = wls_lat[start:end]
                            win_lon = wls_lon[start:end]
                            win_alt = wls_alt[start:end]

                        # Center is at half_window
                        center_lat = win_lat[half_window]
                        center_lon = win_lon[half_window]
                        center_alt = win_alt[half_window]

                        d_east, d_north = latlon_to_meters_diff(
                            center_lat, center_lon, win_lat, win_lon
                        )
                        d_alt = win_alt - center_alt

                        # Construct window
                        window_data = np.zeros(
                            (Config.WINDOW_SIZE, len(Config.FEATURE_NAMES))
                        )

                        # Fill GNSS+IMU+Vel (Static)
                        gnss_imu_cols = Config.GNSS_FEATURES + Config.IMU_FEATURES
                        # We need to map static_feats columns to window_data columns
                        # static_feats contains everything EXCEPT Rel coords.
                        # Config.FEATURE_NAMES = GNSS + IMU + Rel + Vel
                        # static_feat_names = GNSS + IMU + Vel

                        # We assume static_feats follows the order in static_feat_names
                        # We need to map them to the correct indices in Config.FEATURE_NAMES

                        # Optimization: Pre-calculate indices mapping
                        # But simpler: just fill in order if we are careful.
                        # Config order: [GNSS, IMU, RelCoords, Vel]
                        # static order: [GNSS, IMU, Vel]

                        n_gnss_imu = len(gnss_imu_cols)
                        window_data[:, :n_gnss_imu] = win_static[:, :n_gnss_imu]

                        # Fill Rel Coords
                        rel_idx = n_gnss_imu
                        window_data[:, rel_idx] = d_north
                        window_data[:, rel_idx + 1] = d_east
                        window_data[:, rel_idx + 2] = d_alt

                        # Fill Vel
                        window_data[:, rel_idx + 3 :] = win_static[:, n_gnss_imu:]

                        trip_X.append(window_data)

                        # Store metadata for this prediction
                        meta_list.append(
                            {
                                "tripId": trip,
                                "UnixTimeMillis": req_timestamps[k],
                                "WlsLat": center_lat,
                                "WlsLon": center_lon,
                            }
                        )

            X_test = (
                np.array(X_list) if X_list else np.array([])
            )  # Flattened list of list of arrays? No, X_list is list of arrays.
            # X_list is list of (W, F) arrays. np.array(X_list) -> (N, W, F)
            if len(X_list) > 0:
                X_test = np.stack(X_list)

            test_df_final = pd.DataFrame(meta_list)

            np.save(X_test_path, X_test)
            test_df_final.to_parquet(test_meta_path)

        # Apply Scaling
        with open(Config.SCALER_PATH, "r") as f:
            params = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(params["mean"])
        scaler.scale_ = np.array(params["scale"])

        N, W, F = X_test.shape
        X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N, W, F)

        return GNSSTrajectoryDataset(X_test), test_df_final
