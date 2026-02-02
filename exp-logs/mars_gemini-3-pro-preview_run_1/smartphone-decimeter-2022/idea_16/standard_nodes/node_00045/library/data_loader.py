import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import WGS84Utils


class GNSSPreprocessing:
    @staticmethod
    def align_timestamps(df):
        """
        Aligns raw GNSS timestamps to 1Hz by rounding to the nearest second.
        """
        df["UnixTimeMillis"] = df["utcTimeMillis"].apply(
            lambda x: round(x / 1000) * 1000
        )
        return df

    @staticmethod
    def compute_signal_stats(group):
        """
        Computes mean, std, min, max for Cn0DbHz and SvElevationDegrees.
        """
        stats = {}
        for col in ["Cn0DbHz", "SvElevationDegrees"]:
            stats[f"{col}_mean"] = group[col].mean()
            stats[f"{col}_std"] = group[col].std()
            stats[f"{col}_min"] = group[col].min()
            stats[f"{col}_max"] = group[col].max()
        return pd.Series(stats)

    @staticmethod
    def compute_weighted_centroid(group):
        """
        Computes the Signal-Weighted Centroid of the satellite constellation.
        """
        # Convert Azimuth (deg) and Elevation (deg) to radians
        # Azimuth is clockwise from North.
        # Elevation is angle from horizon.
        az_rad = np.radians(group["SvAzimuthDegrees"])
        el_rad = np.radians(group["SvElevationDegrees"])

        # Calculate unit vectors in local ENU-like frame (x=East, y=North, z=Up)
        # x = cos(el) * sin(az)  (East component)
        # y = cos(el) * cos(az)  (North component)
        # z = sin(el)            (Up component)
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)

        weights = group["Cn0DbHz"]
        total_weight = weights.sum()

        if total_weight == 0:
            return pd.Series(
                {
                    "WeightedCentroid_X": 0.0,
                    "WeightedCentroid_Y": 0.0,
                    "WeightedCentroid_Z": 0.0,
                }
            )

        w_x = (x * weights).sum() / total_weight
        w_y = (y * weights).sum() / total_weight
        w_z = (z * weights).sum() / total_weight

        return pd.Series(
            {
                "WeightedCentroid_X": w_x,
                "WeightedCentroid_Y": w_y,
                "WeightedCentroid_Z": w_z,
            }
        )

    @staticmethod
    def compute_reliability_metrics(group):
        """
        Computes satellite count and aggregated uncertainty.
        """
        return pd.Series(
            {
                "SatCount": len(group),
                "RawPseudorangeUncertaintyMeters_mean": group[
                    "RawPseudorangeUncertaintyMeters"
                ].mean(),
            }
        )

    @staticmethod
    def process_drive(drive_id, phone_name, gnss_path, gt_df=None, is_test=False):
        """
        Processes a single drive: loads GNSS, aggregates features, and aligns with GT (if train).
        """
        full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)
        if not os.path.exists(full_gnss_path):
            print(f"Warning: GNSS file not found: {full_gnss_path}")
            return None

        # Load raw GNSS data
        try:
            gnss_df = pd.read_csv(full_gnss_path)
        except Exception as e:
            print(f"Error reading {full_gnss_path}: {e}")
            return None

        # Align timestamps
        gnss_df = GNSSPreprocessing.align_timestamps(gnss_df)

        # Group by timestamp for aggregation
        grouped = gnss_df.groupby("UnixTimeMillis")

        # 1. Signal Stats
        feat_stats = grouped.apply(GNSSPreprocessing.compute_signal_stats)

        # 2. Geometric Moments
        feat_centroid = grouped.apply(GNSSPreprocessing.compute_weighted_centroid)

        # 3. Reliability
        feat_reliability = grouped.apply(GNSSPreprocessing.compute_reliability_metrics)

        # Combine features
        features = pd.concat(
            [feat_stats, feat_centroid, feat_reliability], axis=1
        ).reset_index()

        # Get WLS baseline from GNSS file (first entry per timestamp is sufficient)
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        # Use mean to be safe against minor variations, though usually constant per epoch
        wls_pos = gnss_df.groupby("UnixTimeMillis")[wls_cols].mean().reset_index()

        # Merge features with WLS baseline
        processed_df = pd.merge(features, wls_pos, on="UnixTimeMillis", how="inner")

        # Add metadata
        processed_df["drive_id"] = drive_id
        processed_df["phone_name"] = phone_name

        if not is_test and gt_df is not None:
            # Merge with Ground Truth
            # Cite debug_lesson_4: Normalize Timestamp Precision Before Merging Time-Series Data
            # We must align GT timestamps to the same 1Hz grid as GNSS data to ensure overlap.
            gt_df = gt_df.copy()
            gt_df["UnixTimeMillis"] = gt_df["UnixTimeMillis"].apply(
                lambda x: round(x / 1000) * 1000
            )

            # Ensure types match
            gt_df["UnixTimeMillis"] = gt_df["UnixTimeMillis"].astype(
                processed_df["UnixTimeMillis"].dtype
            )

            # We need Altitude for ECEF conversion. If missing, fill with 0 (sea level approx)
            if "AltitudeMeters" not in gt_df.columns:
                gt_df["AltitudeMeters"] = 0.0

            merged = pd.merge(
                processed_df,
                gt_df[
                    [
                        "UnixTimeMillis",
                        "LatitudeDegrees",
                        "LongitudeDegrees",
                        "AltitudeMeters",
                    ]
                ],
                on="UnixTimeMillis",
                how="inner",
            )

            # Compute Targets
            def compute_targets(row):
                # WLS ECEF
                wls_x, wls_y, wls_z = (
                    row["WlsPositionXEcefMeters"],
                    row["WlsPositionYEcefMeters"],
                    row["WlsPositionZEcefMeters"],
                )

                # GT Geodetic -> ECEF
                gt_lat, gt_lon = row["LatitudeDegrees"], row["LongitudeDegrees"]
                gt_alt = (
                    row["AltitudeMeters"]
                    if not np.isnan(row["AltitudeMeters"])
                    else 0.0
                )

                gt_x, gt_y, gt_z = WGS84Utils.geodetic_to_ecef(gt_lat, gt_lon, gt_alt)

                # We need a local reference frame. Use WLS position as origin.
                # Convert WLS ECEF to Geodetic for the rotation matrix
                ref_lat, ref_lon, ref_alt = WGS84Utils.ecef_to_geodetic(
                    wls_x, wls_y, wls_z
                )

                # Vector from WLS to GT in ENU
                d_e, d_n, d_u = WGS84Utils.ecef_to_enu(
                    gt_x, gt_y, gt_z, ref_lat, ref_lon, ref_alt
                )

                return pd.Series({"dNorth_meters": d_n, "dEast_meters": d_e})

            targets = merged.apply(compute_targets, axis=1)
            merged = pd.concat([merged, targets], axis=1)
            return merged

        else:
            # Test mode: no targets
            return processed_df


def load_dataset(split="train", load_cached_data=True, debug=False):
    """
    Loads and preprocesses the dataset for the given split.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split}_processed.parquet")

    if debug:
        cache_file = cache_file.replace(".parquet", "_debug.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        df = pd.read_parquet(cache_file)
        # Check if cache is stale (missing new features)
        missing_cols = [c for c in Config.FEATURE_COLS if c not in df.columns]
        if not missing_cols:
            return df
        print(f"Cached data missing columns: {missing_cols}. Reprocessing...")

    print(f"Processing {split} data...")

    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    meta_df = pd.read_csv(meta_path)

    if debug:
        meta_df = meta_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Group by drive and phone
    groups = meta_df.groupby(["drive_id", "phone_name"])

    results = []

    for (drive_id, phone_name), group in groups:
        gnss_path = group.iloc[0]["gnss_path"]

        if split != "test":
            # For train/val, we pass the ground truth subset
            processed = GNSSPreprocessing.process_drive(
                drive_id, phone_name, gnss_path, gt_df=group, is_test=False
            )
        else:
            # For test, no GT
            processed = GNSSPreprocessing.process_drive(
                drive_id, phone_name, gnss_path, gt_df=None, is_test=True
            )

        if processed is not None and not processed.empty:
            results.append(processed)

    if not results:
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)

    # Fill any NaNs created during feature engineering
    final_df = final_df.fillna(0)

    # Cache the result
    print(f"Saving {split} data to cache...")
    final_df.to_parquet(cache_file, index=False)

    return final_df


class GNSSSequenceDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        Args:
            df: Preprocessed DataFrame containing features and targets (if train/val).
            mode: 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.feature_cols = Config.FEATURE_COLS
        self.target_cols = Config.TARGET_COLS
        self.baseline_cols = Config.BASELINE_COLS
        self.max_len = Config.MAX_SEQUENCE_LENGTH

        # Group data by sequence (drive_id, phone_name)
        # Sort by time to ensure sequence order
        df = df.sort_values(["drive_id", "phone_name", "UnixTimeMillis"])

        self.sequences = []
        self.metadata = []

        groups = df.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group in groups:
            # Extract features
            feats = group[self.feature_cols].values.astype(np.float32)

            # Extract baseline positions (needed for test inference reconstruction)
            baselines = group[self.baseline_cols].values.astype(np.float64)

            # Extract timestamps
            timestamps = group["UnixTimeMillis"].values

            seq_data = {
                "features": feats,
                "baselines": baselines,
                "timestamps": timestamps,
            }

            if self.mode != "test":
                targets = group[self.target_cols].values.astype(np.float32)
                seq_data["targets"] = targets

            self.sequences.append(seq_data)
            self.metadata.append((drive_id, phone_name))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]

        features = seq_data["features"]  # (L, C_in)
        seq_len = features.shape[0]

        # Pad or truncate
        if seq_len > self.max_len:
            # Truncate (take first max_len)
            features = features[: self.max_len]
            if self.mode != "test":
                targets = seq_data["targets"][: self.max_len]
            baselines = seq_data["baselines"][: self.max_len]
            timestamps = seq_data["timestamps"][: self.max_len]
            mask = np.ones(self.max_len, dtype=np.float32)
        else:
            # Pad
            pad_len = self.max_len - seq_len
            features = np.pad(
                features, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )
            if self.mode != "test":
                targets = seq_data["targets"]
                targets = np.pad(
                    targets, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
                )

            # Pad baselines with 0 (not used in loss, just for shape consistency)
            baselines = np.pad(
                seq_data["baselines"],
                ((0, pad_len), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            # Pad timestamps
            timestamps = np.pad(
                seq_data["timestamps"], (0, pad_len), mode="constant", constant_values=0
            )

            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)]).astype(
                np.float32
            )

        # Transpose to (C, L) for PyTorch Conv1d
        features = torch.tensor(features).permute(1, 0)  # (C_in, L)
        mask = torch.tensor(mask)  # (L)

        item = {
            "features": features,
            "mask": mask,
            "baselines": torch.tensor(baselines),  # (L, 3)
            "timestamps": torch.tensor(timestamps),  # (L)
        }

        if self.mode != "test":
            targets = torch.tensor(targets).permute(1, 0)  # (C_out, L)
            item["targets"] = targets

        return item
