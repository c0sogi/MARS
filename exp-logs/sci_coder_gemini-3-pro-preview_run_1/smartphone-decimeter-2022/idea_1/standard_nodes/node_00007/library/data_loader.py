import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.utils import ecef_to_lla

# Constants
CACHE_DIR = "./working/idea_3/"
FEATURE_COLS = [
    "WlsAlt",
    "Cn0DbHz",
    "SvElevationDegrees",
    "SatCount",
    "RawPseudorangeUncertaintyMeters",
]
TARGET_COLS = ["dLat", "dLon"]


def preprocess_drive(
    drive_id,
    phone_name,
    gnss_path,
    gt_path=None,
    output_dir=CACHE_DIR,
    load_cached_data=True,
):
    """
    Reads raw GNSS data, aggregates to epoch level, converts coordinates,
    and optionally merges with ground truth. Caches result to parquet.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, f"{drive_id}_{phone_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # 1. Load GNSS Data
    # Only load necessary columns to save memory
    req_cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Cn0DbHz",
        "SvElevationDegrees",
        "RawPseudorangeUncertaintyMeters",
    ]
    # Check if file exists
    if not os.path.exists(gnss_path):
        # Return empty DF with expected columns if file missing (edge case)
        return pd.DataFrame(columns=["UnixTimeMillis"] + FEATURE_COLS + TARGET_COLS)

    df_gnss = pd.read_csv(gnss_path, usecols=lambda c: c in req_cols)

    # 2. Aggregate to Epoch level
    # WLS position is per-epoch (same for all sats), Signal info varies.
    agg_funcs = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "Cn0DbHz": "mean",
        "SvElevationDegrees": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
    }
    # Apply aggregations
    df_epoch = df_gnss.groupby("utcTimeMillis").agg(agg_funcs)

    # Calculate Satellite Count (Cite solution_lesson_node_00006)
    df_epoch["SatCount"] = df_gnss.groupby("utcTimeMillis").size()

    df_epoch = df_epoch.reset_index()

    # 3. Coordinate Conversion (ECEF -> LLA)
    # Handle potential NaNs in WLS
    df_epoch = df_epoch.dropna(
        subset=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    )

    if len(df_epoch) > 0:
        lats, lons, alts = ecef_to_lla(
            df_epoch["WlsPositionXEcefMeters"].values,
            df_epoch["WlsPositionYEcefMeters"].values,
            df_epoch["WlsPositionZEcefMeters"].values,
        )
        df_epoch["WlsLat"] = lats
        df_epoch["WlsLon"] = lons
        df_epoch["WlsAlt"] = alts
    else:
        df_epoch["WlsLat"] = []
        df_epoch["WlsLon"] = []
        df_epoch["WlsAlt"] = []

    # Rename time column for consistency
    df_epoch.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # Sort by time
    df_epoch = df_epoch.sort_values("UnixTimeMillis").reset_index(drop=True)

    # 4. Merge Ground Truth (if available)
    if gt_path and os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)
        # GT timestamps are usually clean. GNSS might have jitter.
        # Round GNSS time to nearest second for merging?
        # Actually, competition data usually aligns well.
        # Let's try merge_asof or exact merge on rounded time.
        # Rounding to nearest 1000ms is a safe heuristic for this dataset.

        df_epoch["merge_key"] = np.round(df_epoch["UnixTimeMillis"] / 1000.0) * 1000.0
        df_gt["merge_key"] = np.round(df_gt["UnixTimeMillis"] / 1000.0) * 1000.0

        # We want to keep all GNSS epochs, but attach GT where available
        df_merged = pd.merge(
            df_epoch,
            df_gt[["merge_key", "LatitudeDegrees", "LongitudeDegrees"]],
            on="merge_key",
            how="left",
        )

        # Calculate residuals
        df_merged["dLat"] = df_merged["LatitudeDegrees"] - df_merged["WlsLat"]
        df_merged["dLon"] = df_merged["LongitudeDegrees"] - df_merged["WlsLon"]

        # Drop merge key and GT columns (we only need residuals and WLS)
        df_final = df_merged.drop(
            columns=[
                "merge_key",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        )
    else:
        # Test mode or missing GT
        df_final = df_epoch.drop(
            columns=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        )
        df_final["dLat"] = np.nan
        df_final["dLon"] = np.nan

    # 5. Interpolation
    # Fill small gaps to ensure continuity for CNN
    # Limit direction both to avoid extrapolating too far
    df_final = df_final.interpolate(method="linear", limit=5, limit_direction="both")

    # Save
    df_final.to_parquet(cache_file)
    return df_final


class GnssWindowedDataset(Dataset):
    def __init__(
        self, metadata_df, input_dir, window_size=64, mode="train", scaler=None
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing drive_id, phone_name, UnixTimeMillis.
            input_dir (str): Root directory of input data.
            window_size (int): Size of the sliding window.
            mode (str): 'train' or 'test'.
            scaler (StandardScaler): Pre-fitted scaler. If None and mode='train', fits a new one.
        """
        self.metadata = metadata_df.copy()
        self.input_dir = input_dir
        self.window_size = window_size
        self.mode = mode
        self.half_window = window_size // 2

        # Dictionary to store loaded drive dataframes: key=(drive_id, phone_name) -> value=df
        self.drive_data = {}

        # Load all required drives
        unique_drives = self.metadata[
            ["drive_id", "phone_name", "gnss_path"]
        ].drop_duplicates()

        # For scaler fitting
        all_features = []

        for _, row in unique_drives.iterrows():
            drive_id = row["drive_id"]
            phone = row["phone_name"]

            # Determine GT path
            if mode == "train":
                # In train mode, gnss_path is like train/drive/phone/device_gnss.csv
                # GT is at train/drive/phone/ground_truth.csv
                gt_path = os.path.join(
                    input_dir, os.path.dirname(row["gnss_path"]), "ground_truth.csv"
                )
            else:
                gt_path = None

            gnss_full_path = os.path.join(input_dir, row["gnss_path"])

            df = preprocess_drive(drive_id, phone, gnss_full_path, gt_path)

            # Ensure sorting by time
            df = df.sort_values("UnixTimeMillis").reset_index(drop=True)

            self.drive_data[(drive_id, phone)] = df

            if mode == "train" and scaler is None:
                # Collect valid features for scaler fitting
                # Filter out rows where WlsLat is NaN (failed ECEF conversion)
                valid_rows = df.dropna(subset=FEATURE_COLS)
                if not valid_rows.empty:
                    all_features.append(valid_rows[FEATURE_COLS].values)

        # Handle Scaler
        if scaler is None:
            self.scaler = StandardScaler()
            if len(all_features) > 0:
                self.scaler.fit(np.vstack(all_features))
            else:
                # Fallback if no data found (should not happen in valid setup)
                self.scaler.fit(np.zeros((1, len(FEATURE_COLS))))
        else:
            self.scaler = scaler

        # Pre-calculate indices for fast retrieval
        # We need to map each row in metadata to an index in the specific drive DF
        self.sample_indices = []

        for idx, row in self.metadata.iterrows():
            key = (row["drive_id"], row["phone_name"])
            target_time = row["UnixTimeMillis"]

            if key not in self.drive_data:
                # Should not happen if logic above is correct
                continue

            df = self.drive_data[key]

            # Find the row index closest to target_time
            # Since df is sorted, we can use searchsorted
            # But searchsorted works on arrays.
            times = df["UnixTimeMillis"].values
            # Find insertion point
            pos = np.searchsorted(times, target_time)

            # Check if exact match or closest
            # If pos == len(times), it's after end. If pos == 0, it's before start.
            # We want the closest valid index.
            if pos == len(times):
                pos = len(times) - 1
            elif pos > 0:
                # Check if prev is closer
                if abs(times[pos] - target_time) > abs(times[pos - 1] - target_time):
                    pos = pos - 1

            # Store (key, center_index_in_df)
            self.sample_indices.append((key, pos))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        key, center_idx = self.sample_indices[idx]
        df = self.drive_data[key]

        # Calculate window bounds
        start_idx = center_idx - self.half_window
        end_idx = center_idx + self.half_window

        # Handle Padding
        # If indices are out of bounds, we pad with the edge values
        df_len = len(df)

        # Extract indices to grab (clipping to valid range)
        # We create a list of indices, then clamp them.
        indices = np.arange(start_idx, end_idx)
        clamped_indices = np.clip(indices, 0, df_len - 1)

        # Get data
        data_window = df.iloc[clamped_indices]

        # Features
        features_raw = data_window[FEATURE_COLS].values
        # Fill NaNs in features if any (linear interp might miss edges)
        # Simple forward/backward fill via pandas logic done in preprocess,
        # but if NaNs remain (e.g. all empty), fill 0.
        if np.isnan(features_raw).any():
            features_raw = np.nan_to_num(features_raw, nan=0.0)

        features_norm = self.scaler.transform(features_raw)

        # Target (at center index)
        # Note: center_idx comes from the metadata query, which corresponds to the middle of the window
        # In the windowed array 'features_norm', the center is at index `self.half_window`.

        # We need the target from the dataframe at the actual center_idx
        # We use the row from the dataframe directly to ensure we get the ground truth
        # associated with the query timestamp.
        center_row = df.iloc[center_idx]

        target = np.array([center_row["dLat"], center_row["dLon"]], dtype=np.float32)
        # If target is NaN (test mode), it remains NaN.

        # Baseline (WLS) for reconstruction
        baseline = np.array(
            [center_row["WlsLat"], center_row["WlsLon"]], dtype=np.float64
        )

        timestamp = int(center_row["UnixTimeMillis"])

        return {
            "features": torch.tensor(features_norm, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "baseline": torch.tensor(baseline, dtype=torch.float64),
            "timestamp": torch.tensor(timestamp, dtype=torch.long),
        }
