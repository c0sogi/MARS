import os
import numpy as np
import pandas as pd
import warnings
from library.data_loader import load_dataset
from library.utils import wgs84_to_ecef, ecef_to_enu, ecef_to_wgs84

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
CACHE_DIR = "./working/idea_12"


class FeatureEngineer:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _calculate_doppler_residuals(self, gnss_df):
        """
        Calculates Doppler residuals for each satellite signal.
        Residual = |MeasuredRate - TheoreticalRate|
        TheoreticalRate = Dot(SatVelocity, UnitVector_UserToSat)

        The WLS position is used as the user position proxy.
        """
        # Ensure we have necessary columns
        req_cols = [
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "PseudorangeRateMetersPerSecond",
        ]

        # Check if columns exist
        missing_cols = [c for c in req_cols if c not in gnss_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns for Doppler calculation: {missing_cols}")
            gnss_df["DopplerResidual"] = np.nan
            return gnss_df

        # Filter out rows where WLS or Sat info is missing for calculation
        # We perform calculation on valid rows and map back
        valid_mask = gnss_df[req_cols].notna().all(axis=1)

        if not valid_mask.any():
            gnss_df["DopplerResidual"] = np.nan
            return gnss_df

        # Extract valid data
        # Use numpy arrays for speed
        sv_pos_x = gnss_df.loc[valid_mask, "SvPositionXEcefMeters"].values
        sv_pos_y = gnss_df.loc[valid_mask, "SvPositionYEcefMeters"].values
        sv_pos_z = gnss_df.loc[valid_mask, "SvPositionZEcefMeters"].values

        sv_vel_x = gnss_df.loc[valid_mask, "SvVelocityXEcefMetersPerSecond"].values
        sv_vel_y = gnss_df.loc[valid_mask, "SvVelocityYEcefMetersPerSecond"].values
        sv_vel_z = gnss_df.loc[valid_mask, "SvVelocityZEcefMetersPerSecond"].values

        wls_pos_x = gnss_df.loc[valid_mask, "WlsPositionXEcefMeters"].values
        wls_pos_y = gnss_df.loc[valid_mask, "WlsPositionYEcefMeters"].values
        wls_pos_z = gnss_df.loc[valid_mask, "WlsPositionZEcefMeters"].values

        measured_rate = gnss_df.loc[valid_mask, "PseudorangeRateMetersPerSecond"].values

        # User to Sat Vector
        dx = sv_pos_x - wls_pos_x
        dy = sv_pos_y - wls_pos_y
        dz = sv_pos_z - wls_pos_z

        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Avoid division by zero
        dist = np.maximum(dist, 1e-9)

        # Unit vectors (Line of Sight)
        ux = dx / dist
        uy = dy / dist
        uz = dz / dist

        # Theoretical Range Rate (Sat Velocity projected onto LOS)
        # This assumes static user. The residual captures user motion + errors.
        theo_rate = sv_vel_x * ux + sv_vel_y * uy + sv_vel_z * uz

        # Doppler Residual
        # We take the absolute difference to capture the magnitude of "unexpected" motion/error
        residuals = np.abs(measured_rate - theo_rate)

        # Assign back
        gnss_df.loc[valid_mask, "DopplerResidual"] = residuals

        return gnss_df

    def _create_sector_features(self, gnss_df):
        """
        Aggregates GNSS features by azimuthal sectors (NE, SE, SW, NW).
        """
        if "SvAzimuthDegrees" not in gnss_df.columns:
            print("Warning: SvAzimuthDegrees missing. Skipping sector features.")
            return gnss_df[["tripId", "UnixTimeMillis"]].drop_duplicates()

        # Define sectors
        # NE: 0-90, SE: 90-180, SW: 180-270, NW: 270-360
        az = gnss_df["SvAzimuthDegrees"]
        sectors = {
            "NE": (az >= 0) & (az < 90),
            "SE": (az >= 90) & (az < 180),
            "SW": (az >= 180) & (az < 270),
            "NW": (az >= 270) & (az <= 360),
        }

        # Columns to aggregate
        agg_funcs = {"Cn0DbHz": "mean", "Svid": "count", "SvElevationDegrees": "mean"}

        if "DopplerResidual" in gnss_df.columns:
            agg_funcs["DopplerResidual"] = ["mean", "max"]

        # Base dataframe with unique timestamps
        base_df = (
            gnss_df[["tripId", "UnixTimeMillis"]]
            .drop_duplicates()
            .set_index(["tripId", "UnixTimeMillis"])
        )

        feature_dfs = []

        for sector_name, mask in sectors.items():
            sector_data = gnss_df[mask]

            if sector_data.empty:
                continue

            # Group by timestamp
            grouped = sector_data.groupby(["tripId", "UnixTimeMillis"])

            # Aggregate
            agg_df = grouped.agg(agg_funcs)

            # Flatten columns
            new_cols = []
            for col in agg_df.columns:
                feat_name = col[0]
                stat_name = col[1]
                new_cols.append(f"{sector_name}_{feat_name}_{stat_name}")

            agg_df.columns = new_cols
            feature_dfs.append(agg_df)

        # Join all sector features
        if feature_dfs:
            full_features = pd.concat([base_df] + feature_dfs, axis=1)
        else:
            full_features = base_df

        # Fill NaNs
        # Count -> 0, others -> NaN (LightGBM handles NaN)
        for col in full_features.columns:
            if "count" in col:
                full_features[col] = full_features[col].fillna(0)

        return full_features.reset_index()

    def _add_global_features(self, gnss_df, imu_df, feats_df):
        """
        Adds global aggregates and IMU features.
        """
        # Global GNSS Aggregates
        agg_dict = {"Cn0DbHz": ["mean", "std", "max"], "Svid": "count"}
        if "DopplerResidual" in gnss_df.columns:
            agg_dict["DopplerResidual"] = "mean"

        grouped = gnss_df.groupby(["tripId", "UnixTimeMillis"])
        global_agg = grouped.agg(agg_dict)

        # Flatten columns
        global_agg.columns = [f"Global_{c[0]}_{c[1]}" for c in global_agg.columns]

        # Merge Global GNSS
        feats_df = pd.merge(
            feats_df, global_agg, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # IMU Features
        # imu_df is already aggregated to 1Hz in data_loader (mean)
        accel_cols = [
            "MeasurementX_UncalAccel",
            "MeasurementY_UncalAccel",
            "MeasurementZ_UncalAccel",
        ]

        # Check if we have accel columns
        available_accel = [c for c in accel_cols if c in imu_df.columns]

        if len(available_accel) == 3:
            # Calculate magnitude
            imu_df["AccelMag"] = np.sqrt(
                imu_df["MeasurementX_UncalAccel"] ** 2
                + imu_df["MeasurementY_UncalAccel"] ** 2
                + imu_df["MeasurementZ_UncalAccel"] ** 2
            )

            # Select relevant IMU cols
            imu_feats = imu_df[
                ["tripId", "UnixTimeMillis", "AccelMag"] + available_accel
            ]
            feats_df = pd.merge(
                feats_df, imu_feats, on=["tripId", "UnixTimeMillis"], how="left"
            )
        elif not imu_df.empty:
            # Merge whatever IMU columns exist
            cols_to_merge = [
                c for c in imu_df.columns if c not in ["tripId", "UnixTimeMillis"]
            ]
            if cols_to_merge:
                imu_feats = imu_df[["tripId", "UnixTimeMillis"] + cols_to_merge]
                feats_df = pd.merge(
                    feats_df, imu_feats, on=["tripId", "UnixTimeMillis"], how="left"
                )

        return feats_df

    def _compute_targets(self, feats_df, gt_df, gnss_df):
        """
        Computes ENU residuals (GT - WLS).
        """
        # We need WLS positions for the timestamps in feats_df
        # Get WLS from gnss_df (take first entry per timestamp as WLS is per-epoch)
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        if not all(c in gnss_df.columns for c in wls_cols):
            print("Warning: WLS columns missing. Cannot compute targets.")
            return pd.DataFrame()

        wls_ref = (
            gnss_df.groupby(["tripId", "UnixTimeMillis"])[wls_cols]
            .first()
            .reset_index()
        )

        # Cite debug_lesson_2: Sanitize Regression Targets
        # Drop rows where WLS baseline is missing to prevent NaN targets
        wls_ref = wls_ref.dropna(subset=wls_cols)

        # Merge WLS and GT
        # Use inner join to ensure we have both WLS and GT
        target_df = pd.merge(
            feats_df[["tripId", "UnixTimeMillis"]],
            wls_ref,
            on=["tripId", "UnixTimeMillis"],
            how="inner",
        )
        target_df = pd.merge(
            target_df,
            gt_df[
                [
                    "tripId",
                    "UnixTimeMillis",
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "AltitudeMeters",
                ]
            ],
            on=["tripId", "UnixTimeMillis"],
            how="inner",
        )

        # Fill missing altitude with 0 (affects conversion slightly but acceptable for horizontal error)
        target_df["AltitudeMeters"] = target_df["AltitudeMeters"].fillna(0)

        # Convert GT Lat/Lon/Alt to ECEF
        gt_x, gt_y, gt_z = wgs84_to_ecef(
            target_df["LatitudeDegrees"].values,
            target_df["LongitudeDegrees"].values,
            target_df["AltitudeMeters"].values,
        )

        # Get WLS Lat/Lon/Alt for reference point in ENU conversion
        wls_lat, wls_lon, wls_alt = ecef_to_wgs84(
            target_df["WlsPositionXEcefMeters"].values,
            target_df["WlsPositionYEcefMeters"].values,
            target_df["WlsPositionZEcefMeters"].values,
        )

        # Calculate ENU of GT relative to WLS
        # Vector = GT - WLS
        dE, dN, dU = ecef_to_enu(gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt)

        target_df["target_E"] = dE
        target_df["target_N"] = dN

        return target_df[["tripId", "UnixTimeMillis", "target_E", "target_N"]]

    def create_features(self, split, load_cached_data=True):
        """
        Main pipeline to create features and targets.
        """
        feature_path = os.path.join(CACHE_DIR, f"{split}_features.parquet")
        target_path = os.path.join(CACHE_DIR, f"{split}_targets.parquet")

        if load_cached_data and os.path.exists(feature_path):
            print(f"Loading {split} features from cache...")
            features = pd.read_parquet(feature_path)
            targets = None
            if split in ["train", "val"] and os.path.exists(target_path):
                targets = pd.read_parquet(target_path)
            return features, targets

        print(f"Generating {split} features from scratch...")

        # 1. Load Raw Data
        gnss_df, imu_df, gt_df = load_dataset(split, load_cached_data=True)

        # 2. Calculate Doppler Residuals
        gnss_df = self._calculate_doppler_residuals(gnss_df)

        # 3. Create Sector Features
        feats_df = self._create_sector_features(gnss_df)

        # 4. Add Global & IMU Features
        feats_df = self._add_global_features(gnss_df, imu_df, feats_df)

        # 5. Compute Targets (Train/Val only)
        targets = None
        if split in ["train", "val"] and gt_df is not None:
            targets = self._compute_targets(feats_df, gt_df, gnss_df)

            if not targets.empty:
                # Align features and targets
                merged = pd.merge(
                    feats_df, targets, on=["tripId", "UnixTimeMillis"], how="inner"
                )

                feature_cols = [
                    c for c in feats_df.columns if c not in ["target_E", "target_N"]
                ]
                target_cols = ["tripId", "UnixTimeMillis", "target_E", "target_N"]

                feats_df = merged[feature_cols]
                targets = merged[target_cols]

                print(f"Saving {split} targets to cache...")
                targets.to_parquet(target_path, index=False)
            else:
                print(f"Warning: No targets generated for {split}.")

        # Save Features
        print(f"Saving {split} features to cache...")
        feats_df.to_parquet(feature_path, index=False)

        return feats_df, targets


def process_data(split, load_cached_data=True):
    engineer = FeatureEngineer()
    return engineer.create_features(split, load_cached_data)
