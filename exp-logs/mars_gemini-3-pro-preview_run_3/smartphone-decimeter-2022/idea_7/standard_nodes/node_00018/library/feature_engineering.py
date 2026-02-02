import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import WORKING_DIR, SEED
from library.data_loader import get_train_data, get_val_data, get_test_data
from library.utils import ecef_to_wgs84, ecef_to_enu, wgs84_to_enu


class FeatureEngine:
    def __init__(self):
        self.phone_encoder = LabelEncoder()
        self.is_encoder_fitted = False

    def _aggregate_gnss(self, gnss_df):
        """
        Aggregates raw GNSS data to 1Hz epochs.
        Extracts signal quality features and the WLS baseline position.
        """
        # Create unique trip identifier if not present (safety check)
        if "tripId" not in gnss_df.columns:
            gnss_df["tripId"] = gnss_df["drive_id"] + "-" + gnss_df["phone_name"]

        # Filter out invalid WLS positions (0,0,0) if any
        gnss_df = gnss_df[gnss_df["WlsPositionXEcefMeters"] != 0].copy()

        # Define aggregation dictionary
        agg_dict = {
            "Cn0DbHz": ["mean", "std", "max"],
            "SvElevationDegrees": ["mean"],
            "Svid": ["count"],
            # WLS positions are repeated for every signal in the epoch, take first
            "WlsPositionXEcefMeters": ["first"],
            "WlsPositionYEcefMeters": ["first"],
            "WlsPositionZEcefMeters": ["first"],
        }

        # Group by Trip and Time
        gnss_agg = gnss_df.groupby(["tripId", "utcTimeMillis"]).agg(agg_dict)

        # Flatten columns
        gnss_agg.columns = [
            f"{c[0]}_{c[1]}" if c[1] != "first" else c[0] for c in gnss_agg.columns
        ]

        # Rename for clarity
        gnss_agg = gnss_agg.rename(
            columns={
                "Svid_count": "sv_count",
                "WlsPositionXEcefMeters": "wls_x",
                "WlsPositionYEcefMeters": "wls_y",
                "WlsPositionZEcefMeters": "wls_z",
            }
        )

        gnss_agg = gnss_agg.reset_index()
        gnss_agg = gnss_agg.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

        return gnss_agg

    def _aggregate_imu(self, imu_df):
        """
        Aggregates high-frequency IMU data to 1Hz epochs.
        Calculates acceleration magnitude.
        """
        if imu_df.empty:
            return pd.DataFrame(columns=["tripId", "UnixTimeMillis", "imu_accel_mean"])

        if "tripId" not in imu_df.columns:
            imu_df["tripId"] = imu_df["drive_id"] + "-" + imu_df["phone_name"]

        # Filter for Accelerometer
        accel_df = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()

        if accel_df.empty:
            return pd.DataFrame(columns=["tripId", "UnixTimeMillis", "imu_accel_mean"])

        # Calculate Magnitude
        accel_df["accel_mag"] = np.sqrt(
            accel_df["MeasurementX"] ** 2
            + accel_df["MeasurementY"] ** 2
            + accel_df["MeasurementZ"] ** 2
        )

        # Align timestamps to nearest second (1000ms)
        accel_df["UnixTimeMillis"] = np.round(accel_df["utcTimeMillis"] / 1000) * 1000
        accel_df["UnixTimeMillis"] = accel_df["UnixTimeMillis"].astype(np.int64)

        # Aggregate
        imu_agg = (
            accel_df.groupby(["tripId", "UnixTimeMillis"])["accel_mag"]
            .mean()
            .reset_index()
        )
        imu_agg = imu_agg.rename(columns={"accel_mag": "imu_accel_mean"})

        return imu_agg

    def _compute_kinematics_and_residuals(self, df, is_train=True):
        """
        Computes WLS ENU coordinates, Kinematic features (Speed, Accel),
        and Target Residuals (if training data).
        """
        # Sort to ensure kinematic diffs are correct
        df = df.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

        # 1. Convert WLS ECEF to WLS LLA
        wls_lla = ecef_to_wgs84(
            df["wls_x"].values, df["wls_y"].values, df["wls_z"].values
        )
        df["wls_lat"], df["wls_lon"], df["wls_alt"] = wls_lla

        # 2. Define Anchors (First point of each trip)
        anchors = (
            df.groupby("tripId")
            .first()[["wls_lat", "wls_lon", "wls_alt"]]
            .reset_index()
        )
        anchors.columns = ["tripId", "anchor_lat", "anchor_lon", "anchor_alt"]

        df = pd.merge(df, anchors, on="tripId", how="left")

        # 3. Convert WLS LLA to ENU (relative to anchor)
        wls_enu = wgs84_to_enu(
            df["wls_lat"].values,
            df["wls_lon"].values,
            df["wls_alt"].values,
            df["anchor_lat"].values,
            df["anchor_lon"].values,
            df["anchor_alt"].values,
        )
        df["wls_e"], df["wls_n"], df["wls_u"] = wls_enu

        # 4. Calculate WLS Kinematics (Speed, Accel)
        # Delta time is usually 1s, but we should check
        df["dt"] = df.groupby("tripId")["UnixTimeMillis"].diff() / 1000.0
        df["dt"] = df["dt"].fillna(1.0)  # Fill first row with 1s to avoid div/0

        # Velocity
        df["v_e"] = df.groupby("tripId")["wls_e"].diff().fillna(0) / df["dt"]
        df["v_n"] = df.groupby("tripId")["wls_n"].diff().fillna(0) / df["dt"]
        df["wls_speed"] = np.sqrt(df["v_e"] ** 2 + df["v_n"] ** 2)

        # Acceleration
        df["a_e"] = df.groupby("tripId")["v_e"].diff().fillna(0) / df["dt"]
        df["a_n"] = df.groupby("tripId")["v_n"].diff().fillna(0) / df["dt"]
        df["wls_accel"] = np.sqrt(df["a_e"] ** 2 + df["a_n"] ** 2)

        # 5. Sensor Discrepancy (WLS Accel vs IMU Accel)
        # Note: IMU measures proper acceleration (includes gravity ~9.8).
        # WLS is kinematic. We subtract gravity approximation from IMU or just compare dynamics.
        # Simple heuristic: abs(wls_accel - (imu_accel - 9.8))
        if "imu_accel_mean" in df.columns:
            # Fill missing IMU with gravity (assuming static)
            df["imu_accel_mean"] = df["imu_accel_mean"].fillna(9.8)
            df["accel_discrepancy"] = np.abs(
                df["wls_accel"] - np.abs(df["imu_accel_mean"] - 9.8)
            )
        else:
            df["accel_discrepancy"] = 0.0

        # 6. Calculate Targets (Residuals) if training
        if is_train:
            # We use WLS Altitude as proxy for GT Altitude for ENU conversion
            # This minimizes vertical error impact on horizontal projection
            gt_enu = wgs84_to_enu(
                df["LatitudeDegrees"].values,
                df["LongitudeDegrees"].values,
                df["wls_alt"].values,  # Proxy
                df["anchor_lat"].values,
                df["anchor_lon"].values,
                df["anchor_alt"].values,
            )
            gt_e, gt_n, _ = gt_enu

            df["target_e"] = gt_e - df["wls_e"]
            df["target_n"] = gt_n - df["wls_n"]

        return df

    def preprocess(self, split: str, load_cached_data: bool = True):
        """
        Main pipeline to load, process, and return features/targets.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached intermediate files.

        Returns:
            X (pd.DataFrame): Features
            y (pd.DataFrame or None): Targets (dE, dN) if split != test
            meta (pd.DataFrame): Metadata (tripId, timestamp, etc.)
        """
        cache_path = os.path.join(WORKING_DIR, f"features_{split}.parquet")

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{split.upper()}] Loading features from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print(f"[{split.upper()}] Computing features from scratch...")

            # 1. Load Data
            if split == "train":
                gnss, imu, meta = get_train_data(load_cached_data)
            elif split == "val":
                gnss, imu, meta = get_val_data(load_cached_data)
            else:
                gnss, imu, meta = get_test_data(load_cached_data)

            # 2. Aggregate Sensor Data
            gnss_agg = self._aggregate_gnss(gnss)
            imu_agg = self._aggregate_imu(imu)

            # 3. Merge
            # Left join on metadata to ensure we keep all target rows
            df = pd.merge(meta, gnss_agg, on=["tripId", "UnixTimeMillis"], how="left")
            df = pd.merge(df, imu_agg, on=["tripId", "UnixTimeMillis"], how="left")

            # 4. Compute Kinematics & Residuals
            is_train = split != "test"
            df = self._compute_kinematics_and_residuals(df, is_train=is_train)

            # 5. Encode Phone Name
            if split == "train":
                self.phone_encoder.fit(df["phone_name"].astype(str))
                self.is_encoder_fitted = True

            # Handle unseen labels in val/test by mapping to a default or known class
            # Simple approach: map unknown to -1 or mode, but LabelEncoder crashes.
            # Here we assume phone models in test are seen in train (usually true for this dataset structure)
            # If not, we might need a robust encoder. For now, standard transform.
            if self.is_encoder_fitted:
                # Check for unseen labels
                known_labels = set(self.phone_encoder.classes_)
                df["phone_name"] = df["phone_name"].astype(str)
                # Map unknown to the first class (arbitrary but safe)
                df.loc[~df["phone_name"].isin(known_labels), "phone_name"] = (
                    self.phone_encoder.classes_[0]
                )
                df["phone_idx"] = self.phone_encoder.transform(df["phone_name"])
            else:
                # Fallback if not fitted (shouldn't happen if train runs first)
                df["phone_idx"] = 0

            # 6. Save to Cache
            print(f"[{split.upper()}] Saving features to cache: {cache_path}")
            df.to_parquet(cache_path, index=False)

        # Select Features
        feature_cols = [
            "Cn0DbHz_mean",
            "Cn0DbHz_std",
            "Cn0DbHz_max",
            "SvElevationDegrees_mean",
            "sv_count",
            "wls_speed",
            "wls_accel",
            "imu_accel_mean",
            "accel_discrepancy",
            "phone_idx",
        ]

        # Fill NaNs in features (e.g. missing IMU or GNSS gaps)
        X = df[feature_cols].fillna(0)

        meta_cols = [
            "tripId",
            "UnixTimeMillis",
            "drive_id",
            "phone_name",
            "wls_lat",
            "wls_lon",
            "wls_alt",
            "wls_e",
            "wls_n",
            "wls_u",
            "anchor_lat",
            "anchor_lon",
            "anchor_alt",
        ]
        if split != "test":
            meta_cols += ["LatitudeDegrees", "LongitudeDegrees"]  # GT

        meta = df[meta_cols].copy()

        if split != "test":
            y = df[["target_e", "target_n"]].fillna(0)
            return X, y, meta
        else:
            return X, None, meta
