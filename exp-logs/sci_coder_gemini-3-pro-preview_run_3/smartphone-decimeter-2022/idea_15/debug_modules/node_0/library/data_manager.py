import os
import numpy as np
import pandas as pd
from library.utils import ecef_to_geodetic, geodetic_to_ecef, ecef_to_enu
from library.gnss_math import process_gnss_data


class DataManager:
    def __init__(
        self,
        input_dir="./input",
        metadata_dir="./metadata",
        cache_dir="./working/idea_15",
    ):
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_metadata_path(self, split):
        return os.path.join(self.metadata_dir, f"{split}_metadata.csv")

    def _get_cache_path(self, split):
        return os.path.join(self.cache_dir, f"{split}_dataset.parquet")

    def _load_imu(self, path):
        """Loads and aggregates IMU data to 1Hz."""
        full_path = os.path.join(self.input_dir, path)
        if not os.path.exists(full_path):
            return None

        df_imu = pd.read_csv(full_path)

        # Align to nearest second (1Hz)
        df_imu["UnixTimeMillis"] = np.round(df_imu["utcTimeMillis"] / 1000.0) * 1000.0
        df_imu["UnixTimeMillis"] = df_imu["UnixTimeMillis"].astype(np.int64)

        # Pivot or separate by MessageType
        # We want mean/std of Accel, Gyro, Mag
        # Filter relevant types just in case
        valid_types = ["UncalAccel", "UncalGyro", "UncalMag"]
        df_imu = df_imu[df_imu["MessageType"].isin(valid_types)]

        # Group by Timestamp and Type
        agg_funcs = ["mean", "std"]
        df_agg = df_imu.groupby(["UnixTimeMillis", "MessageType"])[
            ["MeasurementX", "MeasurementY", "MeasurementZ"]
        ].agg(agg_funcs)

        # Flatten columns
        df_agg.columns = [
            f"{t}_{m}_{c}" for m, c in df_agg.columns for t in [m]
        ]  # e.g. MeasurementX_mean
        # This creates names like MeasurementX_mean, but we need to distinguish sensor types.
        # The groupby index is (Time, Type). We unstack Type.

        df_flat = df_agg.unstack(level="MessageType")
        # Now columns are (MeasurementX_mean, UncalAccel), etc.
        df_flat.columns = [
            f"{sensor}_{metric}_{measure}"
            for metric, measure, sensor in df_flat.columns
        ]

        df_flat.reset_index(inplace=True)
        return df_flat

    def _compute_targets(self, df_merged):
        """
        Computes ENU residuals (GT - WLS) as targets.
        Handles missing GT Altitude by using WLS Altitude.
        """
        # WLS ECEF
        wls_x = df_merged["WlsPositionXEcefMeters"].values
        wls_y = df_merged["WlsPositionYEcefMeters"].values
        wls_z = df_merged["WlsPositionZEcefMeters"].values

        # GT Geodetic
        gt_lat = df_merged["LatitudeDegrees"].values
        gt_lon = df_merged["LongitudeDegrees"].values

        # Handle GT Altitude
        # If AltitudeMeters is in GT, use it. Otherwise derive from WLS.
        if "AltitudeMeters" in df_merged.columns:
            gt_alt = df_merged["AltitudeMeters"].values
        else:
            gt_alt = np.full_like(gt_lat, np.nan)

        # Convert WLS ECEF to Geodetic to get fill values for Altitude
        # We iterate or use vectorized function if available.
        # utils.ecef_to_geodetic is scalar/vector compatible if numpy arrays passed?
        # The provided utils.ecef_to_geodetic uses numpy functions, so it should be vectorized.
        wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

        # Fill NaN GT Alt with WLS Alt
        mask_nan = np.isnan(gt_alt)
        gt_alt_filled = gt_alt.copy()
        gt_alt_filled[mask_nan] = wls_alt[mask_nan]

        # Convert GT LLA to ECEF
        gt_x, gt_y, gt_z = geodetic_to_ecef(gt_lat, gt_lon, gt_alt_filled)

        # Compute ENU residuals centered at WLS position
        # Target = GT - WLS (in ENU frame of WLS)
        d_e, d_n, d_u = ecef_to_enu(gt_x, gt_y, gt_z, wls_x, wls_y, wls_z)

        df_merged["target_E"] = d_e
        df_merged["target_N"] = d_n
        df_merged["target_U"] = d_u

        return df_merged

    def _process_drive(self, drive_id, phone_name, df_meta_drive, is_train):
        """
        Processes a single drive: loads GNSS/IMU, computes features, merges with meta/GT.
        """
        # 1. Load GNSS
        # Meta has relative path
        gnss_rel_path = df_meta_drive.iloc[0]["gnss_path"]
        gnss_abs_path = os.path.join(self.input_dir, gnss_rel_path)

        if not os.path.exists(gnss_abs_path):
            print(f"Warning: GNSS file missing for {drive_id} {phone_name}")
            return pd.DataFrame()

        df_gnss = pd.read_csv(gnss_abs_path)

        # 2. Process GNSS (Physics Features)
        # This returns features indexed by utcTimeMillis
        try:
            df_gnss_feats = process_gnss_data(df_gnss)
        except Exception as e:
            print(f"Error processing GNSS for {drive_id} {phone_name}: {e}")
            return pd.DataFrame()

        df_gnss_feats.reset_index(inplace=True)
        df_gnss_feats.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

        # 3. Extract WLS Positions from raw GNSS (one per epoch)
        # We need this for target computation and as a feature
        df_wls = (
            df_gnss.groupby("utcTimeMillis")[
                [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ]
            .first()
            .reset_index()
        )
        df_wls.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

        # Merge WLS back to features
        df_features = pd.merge(df_gnss_feats, df_wls, on="UnixTimeMillis", how="inner")

        # 4. Load and Aggregate IMU
        imu_rel_path = df_meta_drive.iloc[0]["imu_path"]
        df_imu = self._load_imu(imu_rel_path)

        if df_imu is not None:
            df_features = pd.merge(df_features, df_imu, on="UnixTimeMillis", how="left")

        # 5. Merge with Metadata (which contains GT for train/val, or required timestamps for test)
        # The metadata df contains the target timestamps we need.
        # We perform an inner join to keep only the required rows.

        # Rename metadata columns to avoid conflicts if necessary, but they should be clean
        # df_meta_drive has: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees, etc.

        df_merged = pd.merge(
            df_meta_drive, df_features, on="UnixTimeMillis", how="inner"
        )

        # 6. Compute Targets (only for Train/Val)
        if is_train:
            # We need GT Altitude. If it's in metadata (it isn't usually), use it.
            # The metadata generation script didn't include AltitudeMeters in the saved CSV.
            # We must load the original Ground Truth file to get Altitude.
            gt_rel_path = df_meta_drive.iloc[0]["gt_path"]
            gt_abs_path = os.path.join(self.input_dir, gt_rel_path)

            if os.path.exists(gt_abs_path):
                df_gt_full = pd.read_csv(gt_abs_path)
                # We need AltitudeMeters mapped to UnixTimeMillis
                # Check if duplicates exist
                df_gt_full = df_gt_full.drop_duplicates(subset=["UnixTimeMillis"])

                # Merge Altitude into df_merged
                df_merged = pd.merge(
                    df_merged,
                    df_gt_full[["UnixTimeMillis", "AltitudeMeters"]],
                    on="UnixTimeMillis",
                    how="left",
                )

                df_merged = self._compute_targets(df_merged)
            else:
                # If GT file missing (unlikely), can't compute targets
                return pd.DataFrame()

        return df_merged

    def _process_split(self, split, load_cached=True):
        cache_path = self._get_cache_path(split)

        if load_cached and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Processing {split} data from raw files...")
        meta_path = self._get_metadata_path(split)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Group by trip (drive + phone) to process files efficiently
        trips = df_meta.groupby(["drive_id", "phone_name"])

        results = []
        is_train = split in ["train", "val"]

        for (drive_id, phone_name), df_trip_meta in trips:
            # print(f"Processing {drive_id} - {phone_name}")
            df_processed = self._process_drive(
                drive_id, phone_name, df_trip_meta, is_train
            )
            if not df_processed.empty:
                results.append(df_processed)

        if not results:
            raise ValueError(f"No data processed for split {split}")

        df_final = pd.concat(results, ignore_index=True)

        # Save to cache
        print(f"Saving {split} data to cache: {cache_path}")
        df_final.to_parquet(cache_path, index=False)

        return df_final

    def load_train_val(self, load_cached=True, sample_frac=1.0):
        """
        Loads training and validation data.
        Args:
            load_cached (bool): Whether to use cached parquet files.
            sample_frac (float): Fraction of data to return (for debugging).
        """
        train_df = self._process_split("train", load_cached)
        val_df = self._process_split("val", load_cached)

        if sample_frac < 1.0:
            train_df = train_df.sample(frac=sample_frac, random_state=42).reset_index(
                drop=True
            )
            # Keep validation full usually, or sample too
            val_df = val_df.sample(frac=sample_frac, random_state=42).reset_index(
                drop=True
            )

        return train_df, val_df

    def load_test(self, load_cached=True):
        """
        Loads test data.
        """
        test_df = self._process_split("test", load_cached)
        return test_df
