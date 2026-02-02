import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.model import ecef_to_lla


def load_gnss_and_aggregate(drive_id, phone_name):
    """
    Loads raw GNSS data for a specific drive and phone, aggregates it by timestamp.
    """
    # Attempt to locate file in train or test directories
    gnss_path = os.path.join(
        Config.INPUT_DIR, "train", drive_id, phone_name, Config.GNSS_FILENAME
    )
    if not os.path.exists(gnss_path):
        gnss_path = os.path.join(
            Config.INPUT_DIR, "test", drive_id, phone_name, Config.GNSS_FILENAME
        )

    if not os.path.exists(gnss_path):
        print(f"Warning: GNSS file not found for {drive_id} {phone_name}")
        return pd.DataFrame()

    gnss_df = pd.read_csv(gnss_path)

    # Aggregate by epoch (utcTimeMillis)
    # We take the mean of signal metrics and the first valid WLS position estimate
    gnss_agg = (
        gnss_df.groupby("utcTimeMillis")
        .agg(
            {
                "Cn0DbHz": "mean",
                "SvElevationDegrees": "mean",
                "RawPseudorangeUncertaintyMeters": "mean",
                "Svid": "count",
                "WlsPositionXEcefMeters": "first",
                "WlsPositionYEcefMeters": "first",
                "WlsPositionZEcefMeters": "first",
            }
        )
        .reset_index()
    )

    # Rename columns to match Config feature definitions
    gnss_agg.rename(
        columns={
            "Cn0DbHz": Config.FEAT_GNSS_CN0,
            "SvElevationDegrees": Config.FEAT_GNSS_ELEV,
            "RawPseudorangeUncertaintyMeters": Config.FEAT_GNSS_UNCERTAINTY,
            "Svid": Config.FEAT_GNSS_SAT_COUNT,
        },
        inplace=True,
    )

    return gnss_agg


def load_and_process_imu(drive_id, phone_name, gnss_timestamps=None):
    """
    Loads IMU data, computes derived features (magnitude), and aligns to 1Hz timestamps.
    """
    imu_path = os.path.join(
        Config.INPUT_DIR, "train", drive_id, phone_name, Config.IMU_FILENAME
    )
    if not os.path.exists(imu_path):
        imu_path = os.path.join(
            Config.INPUT_DIR, "test", drive_id, phone_name, Config.IMU_FILENAME
        )

    if not os.path.exists(imu_path):
        # Return empty DF with expected columns if file missing
        return pd.DataFrame(
            columns=[
                "TimeBin",
                Config.FEAT_IMU_ACCEL_MEAN,
                Config.FEAT_IMU_ACCEL_STD,
                Config.FEAT_IMU_GYRO_Z_MEAN,
            ]
        )

    imu_df = pd.read_csv(imu_path)

    # 1. Process Accelerometer
    accel = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
    if not accel.empty:
        # Compute magnitude
        accel["mag"] = np.sqrt(
            accel["MeasurementX"] ** 2
            + accel["MeasurementY"] ** 2
            + accel["MeasurementZ"] ** 2
        )
        # Bin by second (align to GNSS utcTimeMillis)
        accel["TimeBin"] = (accel["utcTimeMillis"] // 1000) * 1000
        accel_agg = accel.groupby("TimeBin")["mag"].agg(["mean", "std"]).reset_index()
        accel_agg.rename(
            columns={
                "mean": Config.FEAT_IMU_ACCEL_MEAN,
                "std": Config.FEAT_IMU_ACCEL_STD,
            },
            inplace=True,
        )
    else:
        accel_agg = pd.DataFrame(
            columns=["TimeBin", Config.FEAT_IMU_ACCEL_MEAN, Config.FEAT_IMU_ACCEL_STD]
        )

    # 2. Process Gyroscope
    gyro = imu_df[imu_df["MessageType"] == "UncalGyro"].copy()
    if not gyro.empty:
        gyro["TimeBin"] = (gyro["utcTimeMillis"] // 1000) * 1000
        # Use Z-axis rotation (yaw rate approx)
        gyro_agg = gyro.groupby("TimeBin")["MeasurementZ"].mean().reset_index()
        gyro_agg.rename(
            columns={"MeasurementZ": Config.FEAT_IMU_GYRO_Z_MEAN}, inplace=True
        )
    else:
        gyro_agg = pd.DataFrame(columns=["TimeBin", Config.FEAT_IMU_GYRO_Z_MEAN])

    # Merge IMU features
    imu_agg = pd.merge(accel_agg, gyro_agg, on="TimeBin", how="outer")
    return imu_agg


def compute_baseline_velocity(gnss_df):
    """
    Converts WLS ECEF coordinates to LLA and computes delta (velocity) features.
    """
    if gnss_df.empty:
        return gnss_df

    # Convert ECEF to LLA
    lat, lon, alt = ecef_to_lla(
        gnss_df["WlsPositionXEcefMeters"].values,
        gnss_df["WlsPositionYEcefMeters"].values,
        gnss_df["WlsPositionZEcefMeters"].values,
    )

    gnss_df["WlsLat"] = lat
    gnss_df["WlsLon"] = lon
    gnss_df[Config.FEAT_WLS_ALT] = alt

    # Compute Deltas (Velocity proxy)
    gnss_df[Config.FEAT_WLS_LAT_DELTA] = gnss_df["WlsLat"].diff().fillna(0)
    gnss_df[Config.FEAT_WLS_LON_DELTA] = gnss_df["WlsLon"].diff().fillna(0)

    return gnss_df


def prepare_drive_data(drive_id, phone_name, df_meta_subset, load_cached_data=True):
    """
    Orchestrates the loading, processing, merging, and caching of data for a single drive.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{drive_id}_{phone_name}.parquet")

    # Check cache
    if load_cached_data and os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        # Ensure metadata columns exist (fix for stale cache files)
        if "drive_id" not in df.columns:
            df["drive_id"] = drive_id
        if "phone_name" not in df.columns:
            df["phone_name"] = phone_name
        return df

    # 1. Load GNSS
    gnss_agg = load_gnss_and_aggregate(drive_id, phone_name)
    if gnss_agg.empty:
        return pd.DataFrame()

    # 2. Compute Baseline Info
    gnss_agg = compute_baseline_velocity(gnss_agg)

    # 3. Load IMU
    imu_agg = load_and_process_imu(drive_id, phone_name)

    # 4. Merge GNSS and IMU
    # Create a TimeBin column in GNSS for merging
    gnss_agg["TimeBin"] = (gnss_agg["utcTimeMillis"] // 1000) * 1000

    if not imu_agg.empty:
        merged_df = pd.merge(gnss_agg, imu_agg, on="TimeBin", how="left")

        # Fill missing IMU values with reasonable defaults (gravity for accel, 0 for gyro)
        merged_df[Config.FEAT_IMU_ACCEL_MEAN] = merged_df[
            Config.FEAT_IMU_ACCEL_MEAN
        ].fillna(9.8)
        merged_df[Config.FEAT_IMU_ACCEL_STD] = merged_df[
            Config.FEAT_IMU_ACCEL_STD
        ].fillna(0.0)
        merged_df[Config.FEAT_IMU_GYRO_Z_MEAN] = merged_df[
            Config.FEAT_IMU_GYRO_Z_MEAN
        ].fillna(0.0)
    else:
        merged_df = gnss_agg
        merged_df[Config.FEAT_IMU_ACCEL_MEAN] = 9.8
        merged_df[Config.FEAT_IMU_ACCEL_STD] = 0.0
        merged_df[Config.FEAT_IMU_GYRO_Z_MEAN] = 0.0

    merged_df.drop(columns=["TimeBin"], inplace=True)

    # 5. Merge with Metadata (Targets or TripId)
    # df_meta_subset contains the 'labels' or 'submission targets'.
    # We use a LEFT JOIN to preserve the full time history of the drive for the TCN window,
    # even if some timestamps don't have targets.

    meta_copy = df_meta_subset.copy()
    if "UnixTimeMillis" in meta_copy.columns:
        meta_copy.rename(columns={"UnixTimeMillis": "utcTimeMillis"}, inplace=True)

    if "LatitudeDegrees" in meta_copy.columns:
        # Train/Val case: Join targets
        final_df = pd.merge(
            merged_df,
            meta_copy[["utcTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            on="utcTimeMillis",
            how="left",
        )
        # Compute Residual Targets
        final_df[Config.TARGET_LAT_RES] = (
            final_df["LatitudeDegrees"] - final_df["WlsLat"]
        )
        final_df[Config.TARGET_LON_RES] = (
            final_df["LongitudeDegrees"] - final_df["WlsLon"]
        )
    else:
        # Test case: Join tripId to identify rows that need prediction
        final_df = pd.merge(
            merged_df,
            meta_copy[["utcTimeMillis", "tripId"]],
            on="utcTimeMillis",
            how="left",
        )
        final_df[Config.TARGET_LAT_RES] = np.nan
        final_df[Config.TARGET_LON_RES] = np.nan

    final_df["drive_id"] = drive_id
    final_df["phone_name"] = phone_name

    # Save to cache
    final_df.to_parquet(cache_file)

    return final_df


class SmartphoneDataset(Dataset):
    def __init__(self, metadata_path, window_size, mode="train"):
        self.window_size = window_size
        self.mode = mode

        # Load metadata
        self.meta_df = pd.read_csv(metadata_path)

        # Process all drives listed in metadata
        self.data_frames = []
        unique_drives = self.meta_df[["drive_id", "phone_name"]].drop_duplicates()

        for _, row in unique_drives.iterrows():
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]

            # Filter metadata for this specific drive
            subset = self.meta_df[
                (self.meta_df["drive_id"] == drive_id)
                & (self.meta_df["phone_name"] == phone_name)
            ]

            # Load/Process data
            df = prepare_drive_data(drive_id, phone_name, subset, load_cached_data=True)
            if not df.empty:
                self.data_frames.append(df)

        if self.data_frames:
            self.full_df = pd.concat(self.data_frames, ignore_index=True)
            # Sort to ensure time continuity for windowing
            self.full_df.sort_values(
                ["drive_id", "phone_name", "utcTimeMillis"], inplace=True
            )
            self.full_df.reset_index(drop=True, inplace=True)
        else:
            self.full_df = pd.DataFrame()

        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        if self.full_df.empty:
            return indices

        # Group by drive to ensure windows don't cross drive boundaries
        groups = self.full_df.groupby(["drive_id", "phone_name"])
        for _, group in groups:
            group_indices = group.index.values

            # Identify valid targets (rows that have a label or are required for submission)
            if self.mode == "test":
                is_target = ~group["tripId"].isna()
            else:
                is_target = ~group[Config.TARGET_LAT_RES].isna()

            # Create sliding windows
            for i in range(self.window_size - 1, len(group)):
                # We only add this window if the last element (the prediction point) is a valid target
                if is_target.iloc[i]:
                    indices.append(group_indices[i])

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        df_idx = self.indices[idx]
        start_idx = df_idx - self.window_size + 1
        end_idx = df_idx + 1

        window = self.full_df.iloc[start_idx:end_idx]

        # Extract features and transpose to [Features, Seq_Len] for TCN input
        features = (
            window[Config.INPUT_FEATURES].values.astype(np.float32).transpose(1, 0)
        )

        if self.mode == "test":
            # For test, return features and the WLS baseline (to add residuals to)
            wls_baseline = window.iloc[-1][["WlsLat", "WlsLon"]].values.astype(
                np.float64
            )
            return features, wls_baseline
        else:
            # For train/val, return features and the Residual Targets
            target = window.iloc[-1][
                [Config.TARGET_LAT_RES, Config.TARGET_LON_RES]
            ].values.astype(np.float32)
            return features, target
