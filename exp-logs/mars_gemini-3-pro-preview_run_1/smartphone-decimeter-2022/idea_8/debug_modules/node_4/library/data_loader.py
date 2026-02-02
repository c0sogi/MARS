import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import ecef_to_llh, llh_to_enu


def feature_engineering(gnss_df):
    """
    Aggregates raw GNSS data into signal profiles per timestamp.
    """
    # Temporal Quantization: Round to nearest second
    # utcTimeMillis is in milliseconds.
    gnss_df["UnixTimeMillis"] = (
        (gnss_df["utcTimeMillis"] + 500) // 1000 * 1000
    ).astype(np.int64)

    # Define bins
    cn0_bins = Config.CN0_BINS
    elev_bins = Config.ELEVATION_BINS

    # Pre-calculate bin indices for speed
    gnss_df["cn0_bin"] = pd.cut(
        gnss_df["Cn0DbHz"], bins=cn0_bins, labels=False, include_lowest=True
    )
    gnss_df["elev_bin"] = pd.cut(
        gnss_df["SvElevationDegrees"], bins=elev_bins, labels=False, include_lowest=True
    )

    # Group by timestamp
    grouped = gnss_df.groupby("UnixTimeMillis")

    # 1. Aggregated Stats
    agg_funcs = {
        "Cn0DbHz": ["min", "max", "mean"],
        "SvElevationDegrees": ["min", "max", "mean"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
        "Svid": ["count"],  # Satellite count
    }

    stats_df = grouped.agg(agg_funcs)
    stats_df.columns = ["_".join(col).strip() for col in stats_df.columns.values]
    stats_df.rename(columns={"Svid_count": "SatCount"}, inplace=True)

    # 2. Histogram Counts (Signal Profile)
    def get_bin_counts(col_name, num_bins):
        valid_subset = gnss_df.dropna(subset=[col_name])
        if valid_subset.empty:
            return pd.DataFrame(
                0,
                index=stats_df.index,
                columns=[f"{col_name}_{i}" for i in range(num_bins)],
            )

        counts = pd.crosstab(valid_subset["UnixTimeMillis"], valid_subset[col_name])
        # Reindex to ensure all bins 0..N-1 exist
        counts = counts.reindex(columns=range(num_bins), fill_value=0)
        counts.columns = [f"{col_name}_{i}" for i in range(num_bins)]
        return counts

    cn0_counts = get_bin_counts("cn0_bin", Config.NUM_CN0_BINS)
    elev_counts = get_bin_counts("elev_bin", Config.NUM_ELEV_BINS)

    # Merge all features
    features_df = pd.concat([stats_df, cn0_counts, elev_counts], axis=1).fillna(0)

    # WLS Baseline Extraction (Take the first valid WLS position per second)
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    wls_df = grouped[wls_cols].first()

    final_df = pd.concat([features_df, wls_df], axis=1)

    # Reset index to make UnixTimeMillis a column
    final_df = final_df.reset_index()

    return final_df


def process_drive(drive_id, phone_name, gnss_path, gt_df=None, load_cached_data=True):
    """
    Processes a single drive: loads GNSS, engineers features, aligns with GT (if available).
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{drive_id}_{phone_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache {cache_file}: {e}. Recomputing.")

    # Load Raw Data
    full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)
    if not os.path.exists(full_gnss_path):
        print(f"Warning: GNSS file not found at {full_gnss_path}")
        return pd.DataFrame()  # Return empty

    gnss_df = pd.read_csv(full_gnss_path)

    # Feature Engineering
    processed_df = feature_engineering(gnss_df)

    # Convert WLS ECEF to LLH
    x = processed_df["WlsPositionXEcefMeters"].values
    y = processed_df["WlsPositionYEcefMeters"].values
    z = processed_df["WlsPositionZEcefMeters"].values
    wls_lat, wls_lon, wls_alt = ecef_to_llh(x, y, z)

    processed_df["wls_lat"] = wls_lat
    processed_df["wls_lon"] = wls_lon
    processed_df["wls_alt"] = wls_alt

    # If Ground Truth is provided (Train/Val mode)
    if gt_df is not None:
        gt_df = gt_df.copy()
        # Merge on UnixTimeMillis
        gt_df["UnixTimeMillis"] = gt_df["UnixTimeMillis"].astype(np.int64)

        # Check for missing AltitudeMeters (missing in provided metadata)
        if "AltitudeMeters" not in gt_df.columns:
            raw_gt_path = os.path.join(
                Config.INPUT_DIR, "train", drive_id, phone_name, "ground_truth.csv"
            )
            altitude_loaded = False
            if os.path.exists(raw_gt_path):
                try:
                    raw_gt = pd.read_csv(raw_gt_path)
                    if "AltitudeMeters" in raw_gt.columns:
                        raw_gt["UnixTimeMillis"] = raw_gt["UnixTimeMillis"].astype(
                            np.int64
                        )
                        # Merge AltitudeMeters
                        gt_df = pd.merge(
                            gt_df,
                            raw_gt[["UnixTimeMillis", "AltitudeMeters"]],
                            on="UnixTimeMillis",
                            how="left",
                        )
                        altitude_loaded = True
                except Exception as e:
                    print(f"Error loading raw GT for altitude: {e}")

            # Fallback if still missing
            if not altitude_loaded:
                print(
                    f"Warning: AltitudeMeters missing for {drive_id}-{phone_name}, filling with 0."
                )
                gt_df["AltitudeMeters"] = 0.0

        # Create JoinKey for alignment (Round to nearest second)
        # Cite debug_lesson_4: Normalize Timestamp Precision Before Merging Time-Series Data
        gt_df["JoinKey"] = ((gt_df["UnixTimeMillis"] + 500) // 1000 * 1000).astype(
            np.int64
        )
        processed_df["JoinKey"] = processed_df["UnixTimeMillis"]

        # Inner join to keep only labeled data
        # Rename processed_df time to avoid collision and preserve precise GT time
        merged_df = pd.merge(
            processed_df.rename(columns={"UnixTimeMillis": "UnixTimeMillis_rounded"}),
            gt_df[
                [
                    "JoinKey",
                    "UnixTimeMillis",  # Precise time from GT
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "AltitudeMeters",
                ]
            ],
            on="JoinKey",
            how="inner",
        )

        # Cite debug_lesson_10: Verify Join Key Overlap
        if merged_df.empty:
            print(f"Warning: Empty intersection for {drive_id}-{phone_name}")
            return pd.DataFrame()

        # Calculate Targets (ENU Residuals)
        # Target = GT - WLS (in meters)
        gt_lat = merged_df["LatitudeDegrees"].values
        gt_lon = merged_df["LongitudeDegrees"].values
        gt_alt = merged_df["AltitudeMeters"].values

        ref_lat = merged_df["wls_lat"].values
        ref_lon = merged_df["wls_lon"].values
        ref_alt = merged_df["wls_alt"].values

        e, n, u = llh_to_enu(gt_lat, gt_lon, gt_alt, ref_lat, ref_lon, ref_alt)

        merged_df["target_E"] = e
        merged_df["target_N"] = n

        # Save cache
        merged_df.to_parquet(cache_file)
        return merged_df

    else:
        # Test mode: No GT, just save features
        processed_df.to_parquet(cache_file)
        return processed_df


class GnssSequenceDataset(Dataset):
    def __init__(self, drive_data_list, sequence_length=128, mode="train"):
        """
        Args:
            drive_data_list: List of DataFrames, each representing a processed drive.
            sequence_length: Length of time sequences.
            mode: 'train' or 'test'. If train, returns targets.
        """
        self.mode = mode
        self.sequence_length = sequence_length
        self.samples = []

        # Feature columns (exclude metadata, WLS, targets)
        self.feature_cols = [
            "Cn0DbHz_min",
            "Cn0DbHz_max",
            "Cn0DbHz_mean",
            "SvElevationDegrees_min",
            "SvElevationDegrees_max",
            "SvElevationDegrees_mean",
            "RawPseudorangeUncertaintyMeters_mean",
            "SatCount",
        ]
        # Add bin columns
        self.feature_cols += [f"cn0_bin_{i}" for i in range(Config.NUM_CN0_BINS)]
        self.feature_cols += [f"elev_bin_{i}" for i in range(Config.NUM_ELEV_BINS)]

        # Validate feature count
        assert (
            len(self.feature_cols) == Config.INPUT_CHANNELS
        ), f"Feature count mismatch! Expected {Config.INPUT_CHANNELS}, got {len(self.feature_cols)}"

        # Prepare samples
        self.drive_arrays = []

        for drive_idx, df in enumerate(drive_data_list):
            # Ensure sorted by time
            df = df.sort_values("UnixTimeMillis").reset_index(drop=True)

            # Fill NaNs
            df[self.feature_cols] = df[self.feature_cols].fillna(0)

            features = df[self.feature_cols].values.astype(np.float32)

            targets = None
            wls = None
            meta = None

            if mode == "train":
                targets = df[["target_E", "target_N"]].values.astype(np.float32)

            wls = df[["wls_lat", "wls_lon", "wls_alt"]].values.astype(np.float64)
            meta = df[["UnixTimeMillis"]].values.astype(np.int64)

            num_rows = len(df)

            # If drive is shorter than sequence length, pad it
            if num_rows < sequence_length:
                pad_len = sequence_length - num_rows
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                if targets is not None:
                    targets = np.pad(targets, ((0, pad_len), (0, 0)), mode="constant")
                wls = np.pad(wls, ((0, pad_len), (0, 0)), mode="edge")
                meta = np.pad(meta, ((0, pad_len), (0, 0)), mode="edge")

                # Store as a standalone sample
                self.samples.append(
                    {"features": features, "targets": targets, "wls": wls, "meta": meta}
                )
            else:
                # Store big arrays and use indices
                self.drive_arrays.append(
                    {"features": features, "targets": targets, "wls": wls, "meta": meta}
                )

                # Stride
                stride = 1 if mode == "train" else sequence_length

                for start_idx in range(0, num_rows - sequence_length + 1, stride):
                    self.samples.append((len(self.drive_arrays) - 1, start_idx))

                # Handle remainder for test mode to ensure full coverage
                if mode == "test" and (num_rows - sequence_length) % stride != 0:
                    self.samples.append(
                        (len(self.drive_arrays) - 1, num_rows - sequence_length)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if isinstance(sample, dict):
            # Padded single sample
            features = sample["features"]
            targets = (
                sample["targets"]
                if sample["targets"] is not None
                else np.zeros((self.sequence_length, 2), dtype=np.float32)
            )
            wls = sample["wls"]
            meta = sample["meta"]
        else:
            # Window from drive arrays
            drive_idx, start_idx = sample
            data = self.drive_arrays[drive_idx]
            end_idx = start_idx + self.sequence_length

            features = data["features"][start_idx:end_idx]
            wls = data["wls"][start_idx:end_idx]
            meta = data["meta"][start_idx:end_idx]

            if self.mode == "train":
                targets = data["targets"][start_idx:end_idx]
            else:
                targets = np.zeros((self.sequence_length, 2), dtype=np.float32)

        # Transpose features to (C, L) for PyTorch Conv1d
        features = features.transpose(1, 0)

        # Targets: (L, 2) -> (2, L)
        targets = targets.transpose(1, 0)

        return (
            torch.tensor(features),
            torch.tensor(targets),
            torch.tensor(wls),
            torch.tensor(meta),
        )


def get_train_val_loaders(load_cached_data=True):
    """
    Constructs DataLoaders for training and validation.
    """
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debug mode: sample a fraction of drives
    if Config.DEBUG:
        train_drives_ids = train_meta["drive_id"].unique()[:2]
        train_meta = train_meta[train_meta["drive_id"].isin(train_drives_ids)]
        val_drives_ids = val_meta["drive_id"].unique()[:1]
        val_meta = val_meta[val_meta["drive_id"].isin(val_drives_ids)]

    # Process Training Drives
    train_drives = []
    for (drive_id, phone_name), group in train_meta.groupby(["drive_id", "phone_name"]):
        gnss_path = group.iloc[0]["gnss_path"]
        df = process_drive(
            drive_id,
            phone_name,
            gnss_path,
            gt_df=group,
            load_cached_data=load_cached_data,
        )
        if not df.empty:
            train_drives.append(df)

    # Process Validation Drives
    val_drives = []
    for (drive_id, phone_name), group in val_meta.groupby(["drive_id", "phone_name"]):
        gnss_path = group.iloc[0]["gnss_path"]
        df = process_drive(
            drive_id,
            phone_name,
            gnss_path,
            gt_df=group,
            load_cached_data=load_cached_data,
        )
        if not df.empty:
            val_drives.append(df)

    # Create Datasets
    train_dataset = GnssSequenceDataset(
        train_drives, sequence_length=Config.SEQUENCE_LENGTH, mode="train"
    )
    val_dataset = GnssSequenceDataset(
        val_drives, sequence_length=Config.SEQUENCE_LENGTH, mode="train"
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Constructs DataLoader for test set.
    """
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    test_drives = []
    unique_trips = test_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    for _, row in unique_trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        df = process_drive(
            drive_id,
            phone_name,
            gnss_path,
            gt_df=None,
            load_cached_data=load_cached_data,
        )
        if not df.empty:
            test_drives.append(df)

    test_dataset = GnssSequenceDataset(
        test_drives, sequence_length=Config.SEQUENCE_LENGTH, mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
    )

    return test_loader
