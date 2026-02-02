import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library import config
from library import utils


def get_scaler_stats(df, features, cache_path, load_cached_data=True):
    """
    Compute or load mean and std for features.
    """
    if load_cached_data and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            stats = json.load(f)
        print(f"Loaded scaler stats from {cache_path}")
        return stats

    print("Computing scaler stats...")
    stats = {}
    for col in features:
        # Handle potential NaNs by dropping them for stats calculation
        series = df[col].dropna()
        if len(series) == 0:
            stats[col] = {"mean": 0.0, "std": 1.0}
        else:
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()) + 1e-8,  # Avoid division by zero
            }

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=4)

    return stats


def preprocess_data(metadata_path, mode="train", load_cached_data=True):
    """
    Load, aggregate, feature engineer, and merge data.
    """
    cache_file = config.CACHE_FILES[f"{mode}_data"]

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Processing {mode} data from scratch...")

    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # In Debug mode, sample a subset of trips
    if config.DEBUG:
        trips = df_meta["tripId"].unique()
        sample_trips = trips[: min(len(trips), 5)]
        df_meta = df_meta[df_meta["tripId"].isin(sample_trips)].copy()
        print(f"DEBUG: Sampled {len(sample_trips)} trips.")

    processed_trips = []
    unique_trips = df_meta["tripId"].unique()

    for i, trip_id in enumerate(unique_trips):
        if (i + 1) % 10 == 0:
            print(f"Processing trip {i + 1}/{len(unique_trips)}")

        trip_meta = df_meta[df_meta["tripId"] == trip_id]

        # Get GNSS path from the first row of this trip in metadata
        # The metadata generation script ensures path columns exist
        gnss_rel_path = trip_meta.iloc[0]["gnss_path"]
        gnss_path = os.path.join(config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found for {trip_id}, skipping.")
            continue

        # Load Raw GNSS
        # Only load necessary columns to save memory
        try:
            df_gnss = pd.read_csv(gnss_path, usecols=config.RAW_GNSS_COLS)
        except ValueError as e:
            # Fallback if some columns are missing (e.g. older logs)
            print(f"Warning: Columns missing in {gnss_path}: {e}")
            continue

        # Aggregate by Epoch
        df_gnss_grouped = (
            df_gnss.groupby("utcTimeMillis").agg(config.AGG_MAP).reset_index()
        )
        df_gnss_grouped.rename(columns=config.AGG_RENAME, inplace=True)

        # Convert ECEF to LLA (WLS Baseline)
        wls_lat, wls_lon, wls_alt = utils.ecef_to_lla(
            df_gnss_grouped["WlsPositionXEcefMeters"].values,
            df_gnss_grouped["WlsPositionYEcefMeters"].values,
            df_gnss_grouped["WlsPositionZEcefMeters"].values,
        )

        df_gnss_grouped["WlsLat"] = wls_lat
        df_gnss_grouped["WlsLon"] = wls_lon
        df_gnss_grouped["WlsAlt"] = wls_alt

        # Calculate Dynamics (Deltas)
        # We calculate deltas on the full GNSS sequence to maintain continuity
        df_gnss_grouped["DeltaLat"] = df_gnss_grouped["WlsLat"].diff().fillna(0.0)
        df_gnss_grouped["DeltaLon"] = df_gnss_grouped["WlsLon"].diff().fillna(0.0)
        df_gnss_grouped["DeltaAlt"] = df_gnss_grouped["WlsAlt"].diff().fillna(0.0)

        # Merge with Metadata
        # Train/Val: Inner join (we need GT).
        # Test: Left join (we need to predict for all metadata rows).
        # Note: metadata uses 'UnixTimeMillis', gnss uses 'utcTimeMillis'

        if mode in ["train", "val"]:
            merged = pd.merge(
                trip_meta,
                df_gnss_grouped,
                left_on="UnixTimeMillis",
                right_on="utcTimeMillis",
                how="inner",
            )

            # Calculate Targets (ENU Residuals)
            # Target = GT - WLS
            east, north, up = utils.lla_to_enu(
                merged["LatitudeDegrees"].values,
                merged["LongitudeDegrees"].values,
                merged["AltitudeMeters"].values,
                merged["WlsLat"].values,
                merged["WlsLon"].values,
                merged["WlsAlt"].values,
            )
            merged["DeltaEast"] = east
            merged["DeltaNorth"] = north

            # Sanitize Training Data to Prevent NaN Loss and Logic Failures
            # Cite debug_lesson_2
            merged.dropna(subset=config.INPUT_FEATURES + config.TARGETS, inplace=True)
            merged.reset_index(drop=True, inplace=True)

        else:
            # Test mode
            merged = pd.merge(
                trip_meta,
                df_gnss_grouped,
                left_on="UnixTimeMillis",
                right_on="utcTimeMillis",
                how="left",
            )

            # Handle missing GNSS data in Test (Interpolation)
            # If GNSS is missing for a timestamp required by submission, interpolate features
            feature_cols = config.INPUT_FEATURES
            # Sort by time to ensure interpolation makes sense
            merged = merged.sort_values("UnixTimeMillis")
            merged[feature_cols] = (
                merged[feature_cols]
                .interpolate(method="linear", limit_direction="both")
                .fillna(0)
            )

            # For Test, we don't have targets
            merged["DeltaEast"] = 0.0
            merged["DeltaNorth"] = 0.0

        # Keep only relevant columns
        # Cite debug_lesson_2: Persist Evaluation Metadata Through Feature Selection
        cols_to_keep = (
            [
                "tripId",
                "UnixTimeMillis",
                "phone_name",
                "LatitudeDegrees",
                "LongitudeDegrees",
            ]
            + config.INPUT_FEATURES
            + config.TARGETS
        )
        # Ensure all columns exist
        for c in cols_to_keep:
            if c not in merged.columns:
                merged[c] = 0.0

        processed_trips.append(merged[cols_to_keep])

    if not processed_trips:
        raise ValueError(f"No data processed for mode {mode}")

    full_df = pd.concat(processed_trips, ignore_index=True)

    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    full_df.to_parquet(cache_file, index=False)
    print(f"Saved {mode} data to {cache_file}. Shape: {full_df.shape}")

    return full_df


class GNSSWindowDataset(Dataset):
    def __init__(self, df, window_size, scaler_stats, mode="train"):
        self.window_size = window_size
        self.half_window = window_size // 2
        self.mode = mode
        self.scaler_stats = scaler_stats

        # Features and Targets
        self.feature_cols = config.INPUT_FEATURES
        self.context_cols = config.CONTEXT_FEATURES
        self.target_cols = config.TARGETS

        # Group by trip
        self.trips = []
        self.trip_indices = []

        # Pre-process data into list of numpy arrays per trip
        # We pad each trip to ensure we can generate a window for every point
        print(f"Preparing {mode} dataset windows...")

        grouped = df.groupby("tripId")

        global_idx = 0
        for trip_id, group in grouped:
            # Sort by time just in case
            group = group.sort_values("UnixTimeMillis")

            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)

            # Normalize Features
            for i, col in enumerate(self.feature_cols):
                mean = self.scaler_stats[col]["mean"]
                std = self.scaler_stats[col]["std"]
                features[:, i] = (features[:, i] - mean) / std

            # Pad features
            # We use 'edge' padding (repeat first/last frame)
            # Shape: (N, D) -> (N + 2*half, D)
            padded_features = np.pad(
                features, ((self.half_window, self.half_window), (0, 0)), mode="edge"
            )

            self.trips.append(
                {"features": padded_features, "targets": targets, "length": len(group)}
            )

            # Map global index to (trip_idx, local_idx)
            # local_idx is the index in the original unpadded sequence
            # In the padded sequence, the center is at local_idx + half_window
            for local_idx in range(len(group)):
                self.trip_indices.append((len(self.trips) - 1, local_idx))

    def __len__(self):
        return len(self.trip_indices)

    def __getitem__(self, idx):
        trip_idx, local_idx = self.trip_indices[idx]
        trip_data = self.trips[trip_idx]

        # Window slicing
        # Center of window in padded array is at: local_idx + half_window
        # Window start: (local_idx + half_window) - half_window = local_idx
        # Window end: (local_idx + half_window) + half_window + 1 = local_idx + window_size
        start = local_idx
        end = local_idx + self.window_size

        window_data = trip_data["features"][start:end]  # Shape: (Window, Features)

        # Transpose for 1D CNN: (Features, Window)
        window_tensor = torch.from_numpy(window_data.T)

        # Context: Absolute coordinates of the center frame
        # The center frame in the window is at index `half_window`
        # In the padded array, this corresponds to index `local_idx + half_window`
        center_features = trip_data["features"][local_idx + self.half_window]

        # Extract specific context columns (WlsLat, WlsLon, WlsAlt)
        # We need to know their indices in INPUT_FEATURES
        context_indices = [self.feature_cols.index(c) for c in self.context_cols]
        context_tensor = torch.from_numpy(center_features[context_indices])

        target_tensor = torch.from_numpy(trip_data["targets"][local_idx])

        return window_tensor, context_tensor, target_tensor
