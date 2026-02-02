import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import gc
from library.config import (
    INPUT_DIR,
    WORK_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_TRAIN_FEATURES,
    CACHE_TRAIN_TARGETS,
    CACHE_VAL_FEATURES,
    CACHE_VAL_TARGETS,
    CACHE_TEST_FEATURES,
    TOP_K_SATELLITES,
    SATELLITE_FEATURES,
    GNSS_COLS_TO_LOAD,
    GT_COLS,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import geodetic_to_ecef


class GNSSSnapshotDataset(Dataset):
    """
    PyTorch Dataset for GNSS Snapshot Regression.
    """

    def __init__(self, features, targets=None, meta=None):
        """
        Args:
            features (pd.DataFrame or np.ndarray): Input features (N, Input_Dim).
            targets (pd.DataFrame or np.ndarray, optional): Target residuals (N, 3).
            meta (pd.DataFrame, optional): Metadata for the samples (tripId, timestamp, etc.).
        """
        self.features = (
            torch.tensor(features.values, dtype=torch.float32)
            if isinstance(features, pd.DataFrame)
            else torch.tensor(features, dtype=torch.float32)
        )

        if targets is not None:
            self.targets = (
                torch.tensor(targets.values, dtype=torch.float32)
                if isinstance(targets, pd.DataFrame)
                else torch.tensor(targets, dtype=torch.float32)
            )
        else:
            self.targets = None

        self.meta = meta.reset_index(drop=True) if meta is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


def _process_gnss_group(group):
    """
    Helper to process a group of GNSS measurements for a single timestamp.
    Sorts by signal strength and flattens features.
    """
    # Sort by signal strength descending
    group = group.sort_values("Cn0DbHz", ascending=False)

    # Take top K
    top_k = group.head(TOP_K_SATELLITES)

    # Extract features
    feats = top_k[SATELLITE_FEATURES].values.flatten()

    # Pad if necessary
    expected_len = len(SATELLITE_FEATURES) * TOP_K_SATELLITES
    if len(feats) < expected_len:
        feats = np.pad(feats, (0, expected_len - len(feats)), "constant")

    return feats


def _pivot_gnss_features(df_gnss):
    """
    Vectorized pivoting of GNSS features to create (N, K*F) matrix.
    """
    # Filter to valid signals if needed, but baseline uses all.

    # Sort by timestamp and signal strength
    df_gnss = df_gnss.sort_values(["utcTimeMillis", "Cn0DbHz"], ascending=[True, False])

    # Assign rank within each timestamp
    df_gnss["sat_rank"] = df_gnss.groupby("utcTimeMillis").cumcount()

    # Filter top K
    df_gnss = df_gnss[df_gnss["sat_rank"] < TOP_K_SATELLITES]

    # Pivot
    # Index: utcTimeMillis
    # Columns: sat_rank
    # Values: SATELLITE_FEATURES
    pivot_df = df_gnss.pivot(
        index="utcTimeMillis", columns="sat_rank", values=SATELLITE_FEATURES
    )

    # Flatten columns (MultiIndex to Single Level)
    # New columns will be like: Cn0DbHz_0, Cn0DbHz_1, ...
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]

    # Fill missing satellites with 0
    pivot_df = pivot_df.fillna(0)

    # Ensure all expected columns exist (in case max satellites < K for the whole batch)
    expected_cols = []
    for feat in SATELLITE_FEATURES:
        for k in range(TOP_K_SATELLITES):
            expected_cols.append(f"{feat}_{k}")

    # Reindex to ensure consistent column order and padding
    pivot_df = pivot_df.reindex(columns=expected_cols, fill_value=0)

    return pivot_df


def _load_and_process_split(metadata_path, mode="train"):
    """
    Loads raw data based on metadata and processes it into features and targets.
    """
    print(f"Processing {mode} data from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    if DEBUG_SAMPLE_SIZE:
        print(f"DEBUG: Sampling {DEBUG_SAMPLE_SIZE} rows.")
        meta_df = meta_df.iloc[:DEBUG_SAMPLE_SIZE]

    # We process by drive to manage memory, but we need to collect results
    all_features = []
    all_targets = []
    all_meta = []

    # Group by drive to minimize file I/O (opening GNSS file once per drive)
    for drive_id, group in meta_df.groupby("drive_id"):
        # Get paths from the first row of the group
        first_row = group.iloc[0]

        # Construct absolute paths
        # Metadata paths are relative to input dir
        gnss_path = os.path.join(INPUT_DIR, first_row["gnss_path"])

        # Load GNSS data
        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found: {gnss_path}")
            continue

        # Load only necessary columns to save memory
        try:
            df_gnss = pd.read_csv(gnss_path, usecols=lambda c: c in GNSS_COLS_TO_LOAD)
        except ValueError as e:
            # Handle case where some columns might be missing in specific files
            print(f"Error loading {gnss_path}: {e}")
            continue

        # Filter GNSS to timestamps present in metadata (Ground Truth or Submission)
        required_timestamps = group["UnixTimeMillis"].unique()
        df_gnss = df_gnss[df_gnss["utcTimeMillis"].isin(required_timestamps)]

        if df_gnss.empty:
            continue

        # Pivot features
        features_df = _pivot_gnss_features(df_gnss)

        # Align metadata with processed features
        # features_df index is utcTimeMillis
        # group has UnixTimeMillis

        # We need to extract WLS position for target calculation
        # Take the first WLS position per timestamp (it's repeated for satellites)
        wls_df = df_gnss.groupby("utcTimeMillis")[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].first()

        # Merge features, WLS, and Metadata
        # Metadata acts as the anchor
        merged_df = group.merge(
            features_df, left_on="UnixTimeMillis", right_index=True, how="inner"
        )
        merged_df = merged_df.merge(
            wls_df, left_on="UnixTimeMillis", right_index=True, how="inner"
        )

        # Calculate Targets (only for train/val)
        if mode in ["train", "val"]:
            # Load Ground Truth
            gt_path = os.path.join(INPUT_DIR, first_row["gt_path"])
            if not os.path.exists(gt_path):
                print(f"Warning: GT file not found: {gt_path}")
                continue

            df_gt = pd.read_csv(gt_path, usecols=GT_COLS)

            # Merge GT
            merged_df = merged_df.merge(
                df_gt, on="UnixTimeMillis", how="inner", suffixes=("", "_gt")
            )

            # Convert GT Lat/Lon/Alt to ECEF
            gt_x, gt_y, gt_z = geodetic_to_ecef(
                merged_df["LatitudeDegrees"].values,
                merged_df["LongitudeDegrees"].values,
                merged_df["AltitudeMeters"].values,
            )

            # Calculate Residuals
            target_x = gt_x - merged_df["WlsPositionXEcefMeters"].values
            target_y = gt_y - merged_df["WlsPositionYEcefMeters"].values
            target_z = gt_z - merged_df["WlsPositionZEcefMeters"].values

            targets = np.stack([target_x, target_y, target_z], axis=1)
            all_targets.append(
                pd.DataFrame(targets, columns=["target_x", "target_y", "target_z"])
            )

        # Collect Features (drop non-feature columns)
        feature_cols = features_df.columns.tolist()
        all_features.append(merged_df[feature_cols])

        # Collect Metadata (for ID tracking)
        meta_cols = [
            "tripId",
            "UnixTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        # Add Lat/Lon if available (for reference/debugging)
        if "LatitudeDegrees" in merged_df.columns:
            meta_cols.extend(["LatitudeDegrees", "LongitudeDegrees"])

        all_meta.append(merged_df[meta_cols])

        # Clean up
        del df_gnss, features_df, wls_df, merged_df
        gc.collect()

    if not all_features:
        raise ValueError(f"No data processed for mode {mode}")

    # Concatenate all
    final_features = pd.concat(all_features, ignore_index=True)
    final_meta = pd.concat(all_meta, ignore_index=True)

    if mode in ["train", "val"]:
        final_targets = pd.concat(all_targets, ignore_index=True)
        return final_features, final_targets, final_meta
    else:
        return final_features, None, final_meta


def load_dataset(mode="train", load_cached_data=True):
    """
    Main entry point to load datasets. Handles caching.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        GNSSSnapshotDataset: The ready-to-use PyTorch dataset.
    """
    # Define cache paths based on mode
    if mode == "train":
        meta_path = TRAIN_METADATA_PATH
        feat_cache = CACHE_TRAIN_FEATURES
        target_cache = CACHE_TRAIN_TARGETS
    elif mode == "val":
        meta_path = VAL_METADATA_PATH
        feat_cache = CACHE_VAL_FEATURES
        target_cache = CACHE_VAL_TARGETS
    elif mode == "test":
        meta_path = TEST_METADATA_PATH
        feat_cache = CACHE_TEST_FEATURES
        target_cache = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Try loading cache
    if load_cached_data and os.path.exists(feat_cache):
        print(f"Loading cached {mode} features from {feat_cache}...")
        features = pd.read_parquet(feat_cache)

        targets = None
        if target_cache and os.path.exists(target_cache):
            print(f"Loading cached {mode} targets from {target_cache}...")
            targets = pd.read_parquet(target_cache)

        # We also need metadata for reconstruction/submission, usually we don't cache metadata separately
        # as it is small or we assume index alignment.
        # However, for robustness, we re-load metadata from CSV and filter to match cache length if needed.
        # Ideally, we should cache metadata too to ensure alignment.
        # For simplicity in this implementation, we re-process metadata mapping or assume deterministic order.
        # Let's re-process quickly to get the meta dataframe corresponding to the features.
        # Actually, to avoid re-processing, let's just return the dataset with features/targets.
        # The meta info is critical for Test (WLS positions).
        # We will save meta as parquet as well for Test.
        meta_cache = os.path.join(WORK_DIR, f"{mode}_meta.parquet")
        if os.path.exists(meta_cache):
            meta = pd.read_parquet(meta_cache)
        else:
            # Fallback: re-process if meta cache missing (shouldn't happen if logic is consistent)
            print("Meta cache missing, re-processing data...")
            return load_dataset(mode, load_cached_data=False)

        return GNSSSnapshotDataset(features, targets, meta)

    # Process from scratch
    features, targets, meta = _load_and_process_split(meta_path, mode)

    # Save to cache
    print(f"Saving {mode} features to {feat_cache}...")
    features.to_parquet(feat_cache)

    meta_cache = os.path.join(WORK_DIR, f"{mode}_meta.parquet")
    print(f"Saving {mode} metadata to {meta_cache}...")
    meta.to_parquet(meta_cache)

    if targets is not None and target_cache:
        print(f"Saving {mode} targets to {target_cache}...")
        targets.to_parquet(target_cache)

    return GNSSSnapshotDataset(features, targets, meta)
