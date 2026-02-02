import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from library.utils import EcefToEnu, EcefToGeodetic, WGS84_A, WGS84_E2


class GeometricFeatureExtractor:
    """
    Extracts Split-Band Geometric Projection features and IMU dynamics for GNSS correction.
    """

    def __init__(self, cache_dir="./working/idea_17"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # Signal type classification
        self.l1_signals = {"GPS_L1", "GAL_E1", "GLO_G1", "BDS_B1I", "BDS_B1C", "QZS_J1"}
        self.l5_signals = {"GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"}

    def _get_rotation_matrix(self, lat_deg, lon_deg):
        """
        Computes the ECEF to ENU rotation matrix for a given geodetic position.
        Vectorized for numpy arrays.
        """
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Row 0: East
        r00 = -sin_lon
        r01 = cos_lon
        r02 = np.zeros_like(lon_rad)

        # Row 1: North
        r10 = -sin_lat * cos_lon
        r11 = -sin_lat * sin_lon
        r12 = cos_lat

        # Row 2: Up
        r20 = cos_lat * cos_lon
        r21 = cos_lat * sin_lon
        r22 = sin_lat

        # Stack into (N, 3, 3)
        R = np.stack(
            [
                np.stack([r00, r01, r02], axis=-1),
                np.stack([r10, r11, r12], axis=-1),
                np.stack([r20, r21, r22], axis=-1),
            ],
            axis=1,
        )

        return R

    def _process_drive(self, drive_id, phone_name, gnss_path, imu_path, gt_path=None):
        """
        Process a single drive to extract features and targets.
        """
        # Load Data
        try:
            df_gnss = pd.read_csv(os.path.join("./input", gnss_path))
            df_imu = pd.read_csv(os.path.join("./input", imu_path))
        except FileNotFoundError:
            print(f"Warning: Missing file for {drive_id}-{phone_name}")
            return None

        # 1. Process IMU (Aggregate to 1Hz)
        # IMU data is high frequency, we group by utcTimeMillis
        # We assume utcTimeMillis in IMU is close enough to GNSS epochs
        imu_agg = df_imu.groupby("utcTimeMillis").agg(
            {
                "MeasurementX": ["mean", "std"],
                "MeasurementY": ["mean", "std"],
                "MeasurementZ": ["mean", "std"],
            }
        )
        imu_agg.columns = [f"IMU_{c[0]}_{c[1]}" for c in imu_agg.columns]

        # 2. Process GNSS (Geometric Features)
        # Filter valid signals
        # We need WLS position for every epoch to use as anchor
        # WLS positions are repeated for every satellite in the same epoch

        # Extract unique epochs and their WLS positions
        epoch_cols = [
            "utcTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        df_epochs = df_gnss[epoch_cols].drop_duplicates(subset=["utcTimeMillis"]).copy()

        # Convert WLS ECEF to LLA (Anchor)
        wls_x = df_epochs["WlsPositionXEcefMeters"].values
        wls_y = df_epochs["WlsPositionYEcefMeters"].values
        wls_z = df_epochs["WlsPositionZEcefMeters"].values

        # Handle potential NaNs in WLS
        valid_wls = ~np.isnan(wls_x)

        lat_wls = np.zeros_like(wls_x)
        lon_wls = np.zeros_like(wls_x)
        alt_wls = np.zeros_like(wls_x)

        if np.any(valid_wls):
            lat_v, lon_v, alt_v = EcefToGeodetic.transform(
                wls_x[valid_wls], wls_y[valid_wls], wls_z[valid_wls]
            )
            lat_wls[valid_wls] = lat_v
            lon_wls[valid_wls] = lon_v
            alt_wls[valid_wls] = alt_v

        df_epochs["Wls_Lat"] = lat_wls
        df_epochs["Wls_Lon"] = lon_wls
        df_epochs["Wls_Alt"] = alt_wls

        # Merge Anchor info back to satellites
        df_gnss = df_gnss.merge(
            df_epochs[["utcTimeMillis", "Wls_Lat", "Wls_Lon", "Wls_Alt"]],
            on="utcTimeMillis",
            how="left",
        )

        # Calculate Line-of-Sight (LOS) Vectors in ECEF
        dx = df_gnss["SvPositionXEcefMeters"] - df_gnss["WlsPositionXEcefMeters"]
        dy = df_gnss["SvPositionYEcefMeters"] - df_gnss["WlsPositionYEcefMeters"]
        dz = df_gnss["SvPositionZEcefMeters"] - df_gnss["WlsPositionZEcefMeters"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Unit vectors ECEF
        ux = dx / dist
        uy = dy / dist
        uz = dz / dist

        # Rotate to ENU
        # Get rotation matrices for each row
        R = self._get_rotation_matrix(
            df_gnss["Wls_Lat"].values, df_gnss["Wls_Lon"].values
        )

        # Stack unit vectors: (N, 3)
        u_ecef = np.stack([ux, uy, uz], axis=1)

        # Batch Matrix Multiplication: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
        u_enu = np.einsum("ijk,ik->ij", R, u_ecef)

        df_gnss["u_E"] = u_enu[:, 0]
        df_gnss["u_N"] = u_enu[:, 1]
        df_gnss["u_U"] = u_enu[:, 2]

        # Signal Weighting
        # Convert Cn0 to linear scale roughly proportional to signal power/variance inverse
        df_gnss["weight"] = 10 ** (df_gnss["Cn0DbHz"] / 10.0)

        # Weighted Vectors
        df_gnss["w_uE"] = df_gnss["u_E"] * df_gnss["weight"]
        df_gnss["w_uN"] = df_gnss["u_N"] * df_gnss["weight"]
        df_gnss["w_uU"] = df_gnss["u_U"] * df_gnss["weight"]

        # Assign Bands
        df_gnss["is_L1"] = df_gnss["SignalType"].isin(self.l1_signals)
        df_gnss["is_L5"] = df_gnss["SignalType"].isin(self.l5_signals)

        # Aggregation
        # We need to aggregate by epoch
        # Define aggregation functions

        def compute_band_features(df_band, prefix):
            if df_band.empty:
                return pd.Series(
                    {
                        f"{prefix}_SatCount": 0,
                        f"{prefix}_TotalWeight": 0.0,
                        f"{prefix}_Proj_E": 0.0,
                        f"{prefix}_Proj_N": 0.0,
                        f"{prefix}_Proj_U": 0.0,
                    }
                )

            total_weight = df_band["weight"].sum()
            if total_weight > 0:
                proj_e = df_band["w_uE"].sum() / total_weight
                proj_n = df_band["w_uN"].sum() / total_weight
                proj_u = df_band["w_uU"].sum() / total_weight
            else:
                proj_e, proj_n, proj_u = 0.0, 0.0, 0.0

            return pd.Series(
                {
                    f"{prefix}_SatCount": len(df_band),
                    f"{prefix}_TotalWeight": total_weight,
                    f"{prefix}_Proj_E": proj_e,
                    f"{prefix}_Proj_N": proj_n,
                    f"{prefix}_Proj_U": proj_u,
                }
            )

        # Group and apply is slow. Let's use pivot tables or manual summation for speed.
        # Sums
        grp = df_gnss.groupby("utcTimeMillis")

        # L1 Aggregates
        l1_data = (
            df_gnss[df_gnss["is_L1"]]
            .groupby("utcTimeMillis")[["weight", "w_uE", "w_uN", "w_uU"]]
            .sum()
        )
        l1_counts = df_gnss[df_gnss["is_L1"]].groupby("utcTimeMillis").size()

        # L5 Aggregates
        l5_data = (
            df_gnss[df_gnss["is_L5"]]
            .groupby("utcTimeMillis")[["weight", "w_uE", "w_uN", "w_uU"]]
            .sum()
        )
        l5_counts = df_gnss[df_gnss["is_L5"]].groupby("utcTimeMillis").size()

        # Combine into features dataframe
        features = pd.DataFrame(index=df_epochs["utcTimeMillis"].unique())
        features.index.name = "utcTimeMillis"

        # L1 Features
        features["L1_SatCount"] = l1_counts
        features["L1_TotalWeight"] = l1_data["weight"]
        # Avoid div by zero
        mask_l1 = features["L1_TotalWeight"] > 0
        features.loc[mask_l1, "L1_Proj_E"] = (
            l1_data.loc[mask_l1, "w_uE"] / features.loc[mask_l1, "L1_TotalWeight"]
        )
        features.loc[mask_l1, "L1_Proj_N"] = (
            l1_data.loc[mask_l1, "w_uN"] / features.loc[mask_l1, "L1_TotalWeight"]
        )
        features.loc[mask_l1, "L1_Proj_U"] = (
            l1_data.loc[mask_l1, "w_uU"] / features.loc[mask_l1, "L1_TotalWeight"]
        )

        # L5 Features
        features["L5_SatCount"] = l5_counts
        features["L5_TotalWeight"] = l5_data["weight"]
        mask_l5 = features["L5_TotalWeight"] > 0
        features.loc[mask_l5, "L5_Proj_E"] = (
            l5_data.loc[mask_l5, "w_uE"] / features.loc[mask_l5, "L5_TotalWeight"]
        )
        features.loc[mask_l5, "L5_Proj_N"] = (
            l5_data.loc[mask_l5, "w_uN"] / features.loc[mask_l5, "L5_TotalWeight"]
        )
        features.loc[mask_l5, "L5_Proj_U"] = (
            l5_data.loc[mask_l5, "w_uU"] / features.loc[mask_l5, "L5_TotalWeight"]
        )

        # Fill NaNs (epochs with no L1 or no L5)
        features.fillna(0.0, inplace=True)

        # Merge with WLS info
        features = features.merge(
            df_epochs.set_index("utcTimeMillis"), on="utcTimeMillis", how="left"
        )

        # Merge with IMU
        features = features.merge(imu_agg, on="utcTimeMillis", how="left")

        # 3. Compute Targets (if GT available)
        if gt_path:
            df_gt = pd.read_csv(os.path.join("./input", gt_path))
            # Rename for consistency
            df_gt = df_gt.rename(columns={"UnixTimeMillis": "utcTimeMillis"})

            # Merge GT with features
            # Note: GT might have different timestamps or subset. We typically align to GT timestamps.
            # But here we processed all GNSS epochs. Let's inner join to keep only labeled data.
            features = features.merge(
                df_gt[
                    [
                        "utcTimeMillis",
                        "LatitudeDegrees",
                        "LongitudeDegrees",
                        "AltitudeMeters",
                    ]
                ],
                on="utcTimeMillis",
                how="inner",
                suffixes=("", "_GT"),
            )

            # Compute ENU residuals (Target)
            # Anchor is Wls_Lat, Wls_Lon, Wls_Alt
            # Target is GT Lat/Lon/Alt

            t_e, t_n, t_u = EcefToEnu.transform(
                *GeodeticToEcef.transform(
                    features["LatitudeDegrees"].values,
                    features["LongitudeDegrees"].values,
                    features["AltitudeMeters"].values,
                ),
                features["Wls_Lat"].values,
                features["Wls_Lon"].values,
                features["Wls_Alt"].values,
            )

            features["Target_E"] = t_e
            features["Target_N"] = t_n
            features["Target_U"] = t_u

        # Add ID columns
        features["tripId"] = f"{drive_id}-{phone_name}"
        features["drive_id"] = drive_id
        features["phone_name"] = phone_name

        return features

    def extract_features(self, metadata_df, load_cached_data=True):
        """
        Main method to extract features for all trips in metadata.
        """
        # Identify unique trips
        cols = ["drive_id", "phone_name", "gnss_path", "imu_path"]
        if "gt_path" in metadata_df.columns:
            cols.append("gt_path")

        trips = metadata_df[cols].drop_duplicates()

        all_features = []

        for _, row in tqdm(
            trips.iterrows(), total=len(trips), desc="Extracting Features"
        ):
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]
            trip_id = f"{drive_id}-{phone_name}"

            cache_file = os.path.join(self.cache_dir, f"features_{trip_id}.parquet")

            if load_cached_data and os.path.exists(cache_file):
                df_trip = pd.read_parquet(cache_file)
            else:
                gt_path = (
                    row["gt_path"]
                    if "gt_path" in row and pd.notna(row["gt_path"])
                    else None
                )
                df_trip = self._process_drive(
                    drive_id, phone_name, row["gnss_path"], row["imu_path"], gt_path
                )

                if df_trip is not None:
                    df_trip.to_parquet(cache_file)

            if df_trip is not None:
                all_features.append(df_trip)

        if not all_features:
            return pd.DataFrame()

        return pd.concat(all_features, ignore_index=True)
