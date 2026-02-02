import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.utils import ecef_to_geodetic, geodetic_to_enu

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_11/"
os.makedirs(CACHE_DIR, exist_ok=True)


class GNSSPreprocessor:
    """
    Handles loading and basic cleaning of GNSS logs.
    Aggregates raw measurements to 1Hz epochs.
    """

    def __init__(self):
        pass

    def load_and_clean(self, gnss_path):
        """
        Loads GNSS data, handles timestamp rounding, and extracts WLS baseline.
        """
        full_path = os.path.join(INPUT_DIR, gnss_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"GNSS file not found: {full_path}")

        # Read specific columns to save memory/time
        use_cols = [
            "utcTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "SvAzimuthDegrees",
            "SvElevationDegrees",
            "Cn0DbHz",
            "RawPseudorangeUncertaintyMeters",
        ]

        # Some files might miss columns, handle gracefully?
        # Assuming standard competition format, these should exist.
        df = pd.read_csv(full_path, usecols=lambda c: c in use_cols)

        # Rename for consistency
        df = df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

        # Temporal Quantization: Round to nearest second (1000 ms)
        df["UnixTimeMillis"] = (np.round(df["UnixTimeMillis"] / 1000) * 1000).astype(
            np.int64
        )

        return df

    def extract_baseline(self, df):
        """
        Extracts the WLS baseline position for each epoch.
        Returns a DataFrame with unique timestamps and WLS Lat/Lon/Alt.
        """
        # WLS positions are repeated for every satellite in the same epoch.
        # We take the first one.
        baseline = df[
            [
                "UnixTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].drop_duplicates(subset=["UnixTimeMillis"])

        # Convert ECEF to Geodetic
        x = baseline["WlsPositionXEcefMeters"].values
        y = baseline["WlsPositionYEcefMeters"].values
        z = baseline["WlsPositionZEcefMeters"].values

        lat, lon, alt = ecef_to_geodetic(x, y, z)

        baseline["wls_lat"] = lat
        baseline["wls_lon"] = lon
        baseline["wls_alt"] = alt

        return baseline[["UnixTimeMillis", "wls_lat", "wls_lon", "wls_alt"]]


class SectorFeatureEngineer:
    """
    Computes Sector-Aware features.
    """

    def __init__(self):
        pass

    def create_features(self, df):
        """
        Aggregates satellite data into global and sector-based statistics.
        """
        # 1. Define Sectors (0: NE, 1: SE, 2: SW, 3: NW)
        # Azimuth is 0-360.
        df["sector"] = (df["SvAzimuthDegrees"] // 90).fillna(-1).astype(int)

        # 2. Global Aggregation
        # Group by Timestamp
        g = df.groupby("UnixTimeMillis")

        global_feats = g.agg(
            {
                "Cn0DbHz": ["mean", "max", "min"],
                "SvElevationDegrees": ["mean"],
                "RawPseudorangeUncertaintyMeters": ["mean"],
                "SvAzimuthDegrees": "count",  # Satellite count
            }
        )
        global_feats.columns = [f"global_{c[0]}_{c[1]}" for c in global_feats.columns]
        global_feats = global_feats.rename(
            columns={"global_SvAzimuthDegrees_count": "sat_count"}
        )

        # 3. Sector Aggregation
        # Filter valid sectors
        valid_sectors = df[df["sector"] >= 0]
        s = valid_sectors.groupby(["UnixTimeMillis", "sector"]).agg(
            {"Cn0DbHz": "mean", "SvElevationDegrees": "mean"}
        )

        # Pivot to get (Time x Features)
        # Columns will be like: Cn0DbHz_0, Cn0DbHz_1, ...
        sector_feats = s.unstack(level="sector")
        sector_feats.columns = [f"{c[0]}_{c[1]}" for c in sector_feats.columns]

        # 4. Merge
        features = pd.merge(
            global_feats, sector_feats, left_index=True, right_index=True, how="left"
        )

        # Fill missing sectors with 0 or appropriate default
        features = features.fillna(0)

        return features.reset_index()


class SmartphoneDataset(Dataset):
    """
    PyTorch Dataset for GNSS sequences.
    """

    def __init__(self, processed_data_list, mode="train"):
        """
        Args:
            processed_data_list: List of DataFrames (one per trip).
            mode: 'train' or 'test'.
        """
        self.mode = mode
        self.data = processed_data_list

        # Define feature columns (exclude metadata and targets)
        # We assume all dfs have same columns
        if len(self.data) > 0:
            all_cols = self.data[0].columns.tolist()
            exclude = [
                "UnixTimeMillis",
                "drive_id",
                "phone_name",
                "wls_lat",
                "wls_lon",
                "wls_alt",
                "gt_lat",
                "gt_lon",
                "gt_alt",
                "target_e",
                "target_n",
            ]
            self.feature_cols = [c for c in all_cols if c not in exclude]
        else:
            self.feature_cols = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        df = self.data[idx]

        # Features
        x = df[self.feature_cols].values.astype(np.float32)

        # Metadata for reconstruction
        meta = {
            "UnixTimeMillis": df["UnixTimeMillis"].values,
            "wls_lat": df["wls_lat"].values.astype(np.float64),
            "wls_lon": df["wls_lon"].values.astype(np.float64),
            "wls_alt": df["wls_alt"].values.astype(np.float64),
        }

        if "drive_id" in df.columns:
            meta["drive_id"] = df["drive_id"].iloc[0]
        if "phone_name" in df.columns:
            meta["phone_name"] = df["phone_name"].iloc[0]

        if self.mode == "train":
            # Targets: Offset in meters (East, North)
            y = df[["target_e", "target_n"]].values.astype(np.float32)
            return torch.tensor(x), torch.tensor(y), meta
        else:
            return torch.tensor(x), meta


def process_drive(drive_id, phone_name, gnss_path, gt_path=None, load_cached_data=True):
    """
    Processes a single drive: Loads GNSS, computes features, aligns with GT (if train),
    computes WLS baseline, and caches the result.
    """
    cache_file = os.path.join(CACHE_DIR, f"{drive_id}_{phone_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except:
            pass  # Fallback to re-compute

    # 1. Preprocess GNSS
    preprocessor = GNSSPreprocessor()
    gnss_df = preprocessor.load_and_clean(gnss_path)

    # Extract WLS Baseline (1 row per epoch)
    baseline_df = preprocessor.extract_baseline(gnss_df)

    # 2. Feature Engineering
    engineer = SectorFeatureEngineer()
    features_df = engineer.create_features(gnss_df)

    # Merge Features with Baseline
    # Inner join ensures we only keep epochs where we have both features and WLS
    df = pd.merge(features_df, baseline_df, on="UnixTimeMillis", how="inner")

    # 3. Handle Ground Truth (Training Mode)
    if gt_path:
        full_gt_path = os.path.join(
            INPUT_DIR, gt_path
        )  # gt_path from metadata is relative
        # Note: metadata csv might have absolute or relative. The provided generate_metadata makes them relative.
        # But let's be safe.
        if not os.path.exists(full_gt_path) and os.path.exists(gt_path):
            full_gt_path = gt_path

        if os.path.exists(full_gt_path):
            gt_df = pd.read_csv(full_gt_path)
            # Rename for merge
            gt_df = gt_df.rename(
                columns={
                    "LatitudeDegrees": "gt_lat",
                    "LongitudeDegrees": "gt_lon",
                    "AltitudeMeters": "gt_alt",
                }
            )

            # Merge GT with Features+Baseline
            # Use inner join to align timestamps
            df = pd.merge(
                df,
                gt_df[["UnixTimeMillis", "gt_lat", "gt_lon", "gt_alt"]],
                on="UnixTimeMillis",
                how="inner",
            )

            # Calculate Targets (ENU offsets from WLS to GT)
            # target = GT - WLS
            e, n, u = geodetic_to_enu(
                df["gt_lat"].values,
                df["gt_lon"].values,
                df["gt_alt"].values,
                df["wls_lat"].values,
                df["wls_lon"].values,
                df["wls_alt"].values,
            )
            df["target_e"] = e
            df["target_n"] = n
            # We usually ignore Up for 2D positioning task

    # Add identifiers
    df["drive_id"] = drive_id
    df["phone_name"] = phone_name

    # Save to cache
    df.to_parquet(cache_file, index=False)

    return df


def load_data(metadata_path, split="train", load_cached_data=True, max_drives=None):
    """
    Loads data based on metadata CSV.

    Args:
        metadata_path: Path to train_metadata.csv, val_metadata.csv, or test_metadata.csv.
        split: 'train' or 'test'. Used to determine if we look for GT.
        load_cached_data: Whether to use cached parquet files.
        max_drives: Optional integer to limit number of drives (for debugging).

    Returns:
        SmartphoneDataset containing all processed trips.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Get unique trips (drive + phone)
    # Group by drive_id and phone_name to get unique trips
    trips = meta_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    # For training data, we also need the path to ground truth.
    # The metadata file generated in the description has 'LatitudeDegrees' etc, but not the explicit path to GT file
    # in the columns list shown in the description text (it lists gnss_path, imu_path).
    # However, for training data, the GT path is standard: train/[drive]/[phone]/ground_truth.csv
    # We can reconstruct it.

    processed_dfs = []

    count = 0
    for _, row in trips.iterrows():
        if max_drives is not None and count >= max_drives:
            break

        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        gt_path = None
        if split == "train":
            # Construct GT path
            # Assuming gnss_path is like "train/2020.../Pixel4/device_gnss.csv"
            # GT is "train/2020.../Pixel4/ground_truth.csv"
            gt_path = os.path.dirname(gnss_path) + "/ground_truth.csv"

        try:
            df = process_drive(
                drive_id, phone_name, gnss_path, gt_path, load_cached_data
            )

            # If test, we need to filter to only the timestamps requested in sample_submission
            if split == "test":
                # Filter meta_df for this trip
                req_times = meta_df[
                    (meta_df["drive_id"] == drive_id)
                    & (meta_df["phone_name"] == phone_name)
                ]["UnixTimeMillis"].values

                # Filter processed df
                df = df[df["UnixTimeMillis"].isin(req_times)].copy()

                # Re-sort to match time order (usually already sorted)
                df = df.sort_values("UnixTimeMillis")

            if not df.empty:
                processed_dfs.append(df)
                count += 1

        except Exception as e:
            print(f"Error processing {drive_id} {phone_name}: {e}")
            continue

    return SmartphoneDataset(processed_dfs, mode=split)
