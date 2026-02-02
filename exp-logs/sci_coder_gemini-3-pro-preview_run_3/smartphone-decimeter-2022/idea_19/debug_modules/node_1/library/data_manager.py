import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SEED,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import wgs84_to_ecef, ecef_to_wgs84, ecef_to_enu
from library.feature_builder import build_features


class DataManager:
    def __init__(self, mode="train"):
        self.mode = mode
        self.working_dir = WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _load_metadata(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path)

    def load_train_val_metadata(self):
        """
        Loads training and validation metadata.
        Applies sampling if DEBUG is True.
        """
        train_meta = self._load_metadata(TRAIN_METADATA_PATH)
        val_meta = self._load_metadata(VAL_METADATA_PATH)

        if DEBUG:
            print(f"DEBUG Mode: Sampling {DEBUG_SAMPLE_SIZE} rows for Train/Val.")
            train_meta = train_meta.sample(
                n=min(len(train_meta), DEBUG_SAMPLE_SIZE), random_state=SEED
            )
            val_meta = val_meta.sample(
                n=min(len(val_meta), DEBUG_SAMPLE_SIZE), random_state=SEED
            )

        return train_meta, val_meta

    def load_test_metadata(self):
        """
        Loads test metadata.
        Applies sampling if DEBUG is True.
        """
        test_meta = self._load_metadata(TEST_METADATA_PATH)
        if DEBUG:
            print(f"DEBUG Mode: Sampling {DEBUG_SAMPLE_SIZE} rows for Test.")
            test_meta = test_meta.sample(
                n=min(len(test_meta), DEBUG_SAMPLE_SIZE), random_state=SEED
            )
        return test_meta

    def _compute_targets(self, df):
        """
        Compute ENU targets: Ground Truth Position relative to WLS Position.

        Args:
            df: DataFrame containing merged metadata (with GT) and features (with WLS).

        Returns:
            tuple: (target_E, target_N, target_U) arrays
        """
        # 1. Get WLS ECEF from features
        wls_x = df["WlsPositionXEcefMeters"].values
        wls_y = df["WlsPositionYEcefMeters"].values
        wls_z = df["WlsPositionZEcefMeters"].values

        # 2. Convert WLS to Geodetic (Lat, Lon, Alt) to use as local reference origin
        wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

        # 3. Get GT Geodetic from metadata
        gt_lat = df["LatitudeDegrees"].values
        gt_lon = df["LongitudeDegrees"].values

        # Handle missing Altitude in GT: Fill with WLS altitude
        # This assumes vertical error is zero where GT altitude is missing
        if "AltitudeMeters" in df.columns:
            gt_alt_series = df["AltitudeMeters"]
            gt_alt = np.where(gt_alt_series.isna(), wls_alt, gt_alt_series)
        else:
            gt_alt = wls_alt

        # 4. Convert GT to ECEF
        gt_x, gt_y, gt_z = wgs84_to_ecef(gt_lat, gt_lon, gt_alt)

        # 5. Calculate ENU residuals (Target)
        # Vector from WLS (Origin) to GT
        t_e, t_n, t_u = ecef_to_enu(gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt)

        return t_e, t_n, t_u

    def prepare_dataset(self, metadata_df, split_name, load_cached_data=True):
        """
        Generates or loads the dataset (Features + Targets).

        Args:
            metadata_df: DataFrame containing metadata for the split.
            split_name: 'train', 'val', or 'test'.
            load_cached_data: Whether to load from parquet cache.

        Returns:
            pd.DataFrame: The processed dataset.
        """
        cache_path = os.path.join(self.working_dir, f"{split_name}_dataset.parquet")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split_name} dataset from cache: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache ({e}). Recomputing...")

        print(f"Processing {split_name} dataset from scratch...")

        # Group by drive to process efficiently
        # We process each drive once, then merge with metadata
        unique_drives = metadata_df[
            ["drive_id", "phone_name", "gnss_path"]
        ].drop_duplicates()

        processed_dfs = []

        for _, row in unique_drives.iterrows():
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]
            gnss_rel_path = row["gnss_path"]

            gnss_abs_path = os.path.join(INPUT_DIR, gnss_rel_path)

            if not os.path.exists(gnss_abs_path):
                print(f"Warning: GNSS file not found: {gnss_abs_path}")
                continue

            # Load Raw GNSS
            try:
                gnss_df = pd.read_csv(gnss_abs_path)
                if "utcTimeMillis" in gnss_df.columns:
                    gnss_df.rename(
                        columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True
                    )
            except Exception as e:
                print(f"Error reading {gnss_abs_path}: {e}")
                continue

            # Build Features (This handles its own caching per drive)
            features = build_features(
                drive_id, phone_name, gnss_df, load_cached_data=load_cached_data
            )

            # Filter metadata for this drive
            drive_meta = metadata_df[
                (metadata_df["drive_id"] == drive_id)
                & (metadata_df["phone_name"] == phone_name)
            ].copy()

            # Merge features with metadata on UnixTimeMillis
            # Inner join ensures we only keep epochs present in both (valid features + valid label requirement)
            merged = pd.merge(drive_meta, features, on="UnixTimeMillis", how="inner")

            # Compute Targets if Ground Truth is available (train/val splits)
            # We check for 'LatitudeDegrees' which comes from GT in metadata
            if split_name in ["train", "val"] and "LatitudeDegrees" in merged.columns:
                t_e, t_n, t_u = self._compute_targets(merged)
                merged["target_E"] = t_e
                merged["target_N"] = t_n
                merged["target_U"] = t_u

            processed_dfs.append(merged)

        if not processed_dfs:
            raise ValueError(
                f"No data processed for split {split_name}. Check input paths."
            )

        final_df = pd.concat(processed_dfs, ignore_index=True)

        # Save to cache
        try:
            final_df.to_parquet(cache_path, index=False)
            print(f"Saved {split_name} dataset to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save dataset cache: {e}")

        return final_df

    def get_X_y(self, df, target_col=None):
        """
        Extract features (X) and target (y) from the dataframe.
        Drops metadata columns and absolute position columns to prevent overfitting.

        Args:
            df: Processed dataframe.
            target_col: Name of the target column (e.g., 'target_E'). If None, returns only X.

        Returns:
            X (pd.DataFrame), y (pd.Series) or X (pd.DataFrame)
        """
        # Columns to exclude from features
        exclude_cols = [
            # Metadata
            "tripId",
            "drive_id",
            "phone_name",
            "gnss_path",
            "imu_path",
            "gt_path",
            # Ground Truth
            "LatitudeDegrees",
            "LongitudeDegrees",
            "AltitudeMeters",
            "SpeedMps",
            "AccuracyMeters",
            "BearingDegrees",
            # Index
            "UnixTimeMillis",
            # Targets
            "target_E",
            "target_N",
            "target_U",
            # Absolute WLS Positions (We want translation invariance, using residuals only)
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        # Select feature columns
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]

        if target_col:
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found in dataframe.")
            y = df[target_col]
            return X, y

        return X
