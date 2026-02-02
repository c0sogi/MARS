import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import ecef_to_lla


class GNSSDataset(Dataset):
    """
    PyTorch Dataset for Windowed GNSS Data.
    """

    def __init__(self, features, targets=None, mode="train"):
        self.mode = mode
        self.features = torch.FloatTensor(features)

        if mode in ["train", "val"] and targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.mode in ["train", "val"]:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


def aggregate_gnss(gnss_df):
    """
    Aggregates raw GNSS measurements by timestamp.
    """
    # Define aggregation functions
    agg_funcs = {
        "Svid": "count",
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Filter for columns that actually exist in the dataframe
    existing_cols = gnss_df.columns
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in existing_cols}

    # Group by timestamp
    df_agg = gnss_df.groupby(Config.COL_UTC_TIME).agg(agg_funcs).reset_index()

    # Rename columns to match Config
    rename_map = {
        "Svid": Config.FEAT_SAT_COUNT,
        "Cn0DbHz": Config.FEAT_CN0,
        "RawPseudorangeUncertaintyMeters": Config.FEAT_UNC,
        Config.COL_UTC_TIME: Config.COL_UNIX_TIME,
    }
    df_agg.rename(columns=rename_map, inplace=True)

    return df_agg


def add_derived_features(df):
    """
    Adds derived features like LLA coordinates from ECEF.
    """
    # Convert ECEF to LLA
    if "WlsPositionXEcefMeters" in df.columns:
        lat, lon, alt = ecef_to_lla(
            df["WlsPositionXEcefMeters"].values,
            df["WlsPositionYEcefMeters"].values,
            df["WlsPositionZEcefMeters"].values,
        )
        df[Config.FEAT_WLS_LAT] = lat
        df[Config.FEAT_WLS_LON] = lon
        df[Config.FEAT_WLS_ALT] = alt

        # Drop ECEF columns to save memory
        df.drop(
            columns=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ],
            inplace=True,
        )

    return df


def process_trip(trip_id, gnss_path, gt_df=None):
    """
    Loads and processes a single trip's data.
    """
    full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)

    if not os.path.exists(full_gnss_path):
        print(f"Warning: File not found {full_gnss_path}")
        return pd.DataFrame()

    # Load GNSS data
    try:
        df_gnss = pd.read_csv(full_gnss_path)
    except Exception as e:
        print(f"Error reading {full_gnss_path}: {e}")
        return pd.DataFrame()

    # Aggregate
    df_agg = aggregate_gnss(df_gnss)

    # Add derived features (LLA)
    df_agg = add_derived_features(df_agg)

    # Compute Scaled Deltas (Trajectory Dynamics)
    # Cite solution_lesson_node_00001: Scaling degree differences to roughly meter-scale magnitudes
    df_agg[Config.FEAT_DELTA_LAT] = (
        df_agg[Config.FEAT_WLS_LAT].diff().fillna(0) * Config.TARGET_SCALE_FACTOR
    )
    df_agg[Config.FEAT_DELTA_LON] = (
        df_agg[Config.FEAT_WLS_LON].diff().fillna(0) * Config.TARGET_SCALE_FACTOR
    )

    # Add trip ID
    df_agg[Config.COL_TRIP_ID] = trip_id

    # Merge with Ground Truth if available
    if gt_df is not None:
        # Ground truth timestamps might not align perfectly, but usually they do in this dataset
        # We perform an inner join on timestamps
        df_merged = pd.merge(
            df_agg,
            gt_df[[Config.COL_UNIX_TIME, Config.COL_LATITUDE, Config.COL_LONGITUDE]],
            on=Config.COL_UNIX_TIME,
            how="inner",
        )

        # Compute Residuals
        # Apply scaling to targets (Cite solution_lesson_node_00001)
        df_merged[Config.TARGET_LAT_RES] = (
            df_merged[Config.COL_LATITUDE] - df_merged[Config.FEAT_WLS_LAT]
        ) * Config.TARGET_SCALE_FACTOR
        df_merged[Config.TARGET_LON_RES] = (
            df_merged[Config.COL_LONGITUDE] - df_merged[Config.FEAT_WLS_LON]
        ) * Config.TARGET_SCALE_FACTOR

        # Drop rows where inputs or targets are NaN to prevent training failure
        cols_to_check = Config.INPUT_FEATURES + Config.TARGETS
        # Ensure we only check columns that exist in the dataframe
        cols_to_check = [c for c in cols_to_check if c in df_merged.columns]
        df_merged.dropna(subset=cols_to_check, inplace=True)

        return df_merged
    else:
        # For test set, we keep all aggregated rows.
        # Impute NaNs in features to prevent inference failure
        cols_to_impute = [c for c in Config.INPUT_FEATURES if c in df_agg.columns]
        # Interpolate to fill gaps reasonably
        df_agg[cols_to_impute] = df_agg[cols_to_impute].interpolate(
            method="linear", limit_direction="both"
        )
        # Fill remaining NaNs (e.g. at start/end if interpolate failed) with 0
        df_agg[cols_to_impute] = df_agg[cols_to_impute].fillna(0)

        return df_agg


def create_sliding_windows(df, window_size, features, targets=None):
    """
    Creates flattened sliding window features and corresponding targets.
    """
    # Ensure data is sorted by trip and time
    df = df.sort_values(by=[Config.COL_TRIP_ID, Config.COL_UNIX_TIME]).reset_index(
        drop=True
    )

    feature_data = df[features].values
    trip_ids = df[Config.COL_TRIP_ID].values
    timestamps = df[Config.COL_UNIX_TIME].values

    X_list = []
    y_list = []
    meta_list = []  # To keep track of (tripId, timestamp) for test set reconstruction

    pad_size = window_size // 2

    # We iterate through the dataframe.
    # Since we have multiple trips concatenated, we must ensure windows don't cross trip boundaries.
    # A simple way is to group by trip_id.

    unique_trips = df[Config.COL_TRIP_ID].unique()

    for trip in unique_trips:
        trip_indices = np.where(trip_ids == trip)[0]
        trip_features = feature_data[trip_indices]

        if targets is not None:
            trip_targets = df.loc[trip_indices, targets].values

        trip_timestamps = timestamps[trip_indices]

        n_samples = len(trip_indices)

        if n_samples == 0:
            continue

        # We need to pad the beginning and end of the trip to maintain output size same as input size
        # Padding strategy: Repeat the first/last element
        # Shape: (n_samples, n_features)

        # Create padded array
        padded_features = np.pad(
            trip_features, ((pad_size, pad_size), (0, 0)), mode="edge"
        )

        # Create windows
        # We want a window centered at i. Window range: [i, i + window_size] in padded array
        # corresponding to original index i.

        # Use stride_tricks for efficiency or simple loop
        # Given the dataset size, a simple loop with pre-allocation or list comprehension is fine

        for i in range(n_samples):
            # Window from padded array
            window = padded_features[
                i : i + window_size
            ]  # Shape (window_size, n_features)
            # Flatten
            X_list.append(window.flatten())

            if targets is not None:
                y_list.append(trip_targets[i])

            meta_list.append((trip, trip_timestamps[i]))

    X = np.array(X_list)
    y = np.array(y_list) if targets is not None else None

    return X, y, meta_list


def fit_and_save_scaler(df, features, save_path):
    """
    Computes mean and std for features and saves to JSON.
    """
    stats = {}
    for col in features:
        stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}
        # Avoid division by zero or NaN std
        if stats[col]["std"] == 0 or np.isnan(stats[col]["std"]):
            stats[col]["std"] = 1.0
        if np.isnan(stats[col]["mean"]):
            stats[col]["mean"] = 0.0

    with open(save_path, "w") as f:
        json.dump(stats, f)

    return stats


def load_scaler(load_path):
    """
    Loads scaler stats from JSON.
    """
    with open(load_path, "r") as f:
        stats = json.load(f)
    return stats


def normalize_features(df, features, stats):
    """
    Applies standard scaling to features in the dataframe.
    """
    df_scaled = df.copy()
    for col in features:
        if col in stats:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            df_scaled[col] = (df_scaled[col] - mean) / std
        else:
            # Handle case where feature might be missing from stats (should not happen)
            df_scaled[col] = 0.0

    # Final check for any remaining NaNs after normalization
    df_scaled[features] = df_scaled[features].fillna(0)

    return df_scaled


def get_data(split="train", load_cached_data=True):
    """
    Main function to load and process data for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        tuple: (X, y, meta) for train/val, (X, meta) for test.
    """

    # Determine paths
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_TRAIN_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_VAL_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHE_TEST_PATH
    else:
        raise ValueError("Invalid split")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        # 2. Process from Scratch
        print(f"Processing {split} data from raw files...")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Debugging limit
        if Config.DEBUG_TRIP_COUNT is not None:
            unique_trips = df_meta[Config.COL_TRIP_ID].unique()[
                : Config.DEBUG_TRIP_COUNT
            ]
            df_meta = df_meta[df_meta[Config.COL_TRIP_ID].isin(unique_trips)]
            print(f"Debug mode: Processing only {len(unique_trips)} trips.")

        processed_trips = []
        unique_trips = df_meta[Config.COL_TRIP_ID].unique()

        for trip_id in unique_trips:
            trip_info = df_meta[df_meta[Config.COL_TRIP_ID] == trip_id].iloc[0]
            gnss_path = trip_info["gnss_path"]

            if split in ["train", "val"]:
                # For train/val, we pass the ground truth subset for this trip
                gt_subset = df_meta[df_meta[Config.COL_TRIP_ID] == trip_id]
                df_trip = process_trip(trip_id, gnss_path, gt_subset)
            else:
                # For test, no ground truth
                df_trip = process_trip(trip_id, gnss_path)

            if not df_trip.empty:
                processed_trips.append(df_trip)

        if not processed_trips:
            raise ValueError(f"No data processed for split {split}")

        df = pd.concat(processed_trips, ignore_index=True)

        # Save to cache
        print(f"Saving {split} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

    # 3. Normalization
    # If train, compute and save stats. If val/test, load stats.
    if split == "train":
        print("Computing scaler statistics on training data...")
        scaler_stats = fit_and_save_scaler(
            df, Config.INPUT_FEATURES, Config.SCALER_STATS_PATH
        )
    else:
        if not os.path.exists(Config.SCALER_STATS_PATH):
            raise FileNotFoundError("Scaler stats not found. Run training first.")
        scaler_stats = load_scaler(Config.SCALER_STATS_PATH)

    df_normalized = normalize_features(df, Config.INPUT_FEATURES, scaler_stats)

    # 4. Windowing
    print(f"Creating sliding windows for {split}...")
    targets_cols = Config.TARGETS if split in ["train", "val"] else None

    X, y, meta = create_sliding_windows(
        df_normalized, Config.WINDOW_SIZE, Config.INPUT_FEATURES, targets_cols
    )

    print(f"{split} data ready. Shape: {X.shape}")

    if split in ["train", "val"]:
        return X, y, meta
    else:
        return (
            X,
            meta,
            df,
        )  # Return df for test to access original WLS positions if needed later
