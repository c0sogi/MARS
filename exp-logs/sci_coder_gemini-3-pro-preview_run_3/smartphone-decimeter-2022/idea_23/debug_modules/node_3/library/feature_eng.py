import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_TRAIN_FEATURES,
    CACHE_VAL_FEATURES,
    CACHE_TEST_FEATURES,
    LIGHT_SPEED,
)
from library.coord_utils import wgs84_to_ecef, ecef_to_enu, get_enu_rotation_matrix
from library.data_loader import GnssLoader


class FeatureEngine:
    """
    Implements Stream A: Point-Wise ML Features.
    Computes Unified Geometric Projections (Net Forces) for LightGBM.
    """

    def __init__(self, working_dir=WORKING_DIR):
        self.working_dir = working_dir
        self.loader = GnssLoader(working_dir=working_dir)
        self.cache_dir = os.path.join(working_dir, "features_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_drive_cache_path(self, drive_id, phone_name):
        """Generates a cache path for a specific drive."""
        return os.path.join(self.cache_dir, f"features_{drive_id}_{phone_name}.parquet")

    def compute_los_vectors(self, sat_pos, user_pos):
        """
        Compute Line-of-Sight unit vectors from User to Satellites.
        """
        diff = sat_pos - user_pos
        dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)
        # Avoid division by zero
        dist = np.where(dist < 1e-3, 1e-3, dist)
        u_vec = diff / dist
        return u_vec, dist.flatten()

    def compute_forces(self, residuals, u_vec_enu, weights):
        """
        Compute weighted net force vectors in ENU.
        F = Sum(w * r * u) / Sum(w)
        """
        # residuals: (N,)
        # u_vec_enu: (N, 3)
        # weights: (N,)

        weighted_res = (residuals * weights).reshape(-1, 1)  # (N, 1)
        force_components = weighted_res * u_vec_enu  # (N, 3)

        sum_weights = np.sum(weights)
        if sum_weights < 1e-9:
            return np.zeros(3)

        net_force = np.sum(force_components, axis=0) / sum_weights
        return net_force

    def process_drive(self, drive_id, phone_name, split, load_cached_data=True):
        """
        Computes features for a single drive.
        """
        cache_path = self._get_drive_cache_path(drive_id, phone_name)

        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass  # Fallback to computation if cache read fails

        # Load data
        gnss_df, imu_df, gt_df = self.loader.get_drive_data(
            drive_id, phone_name, split, load_cached_data=load_cached_data
        )

        # Pre-process GNSS
        # Ensure we have WLS positions
        if "WlsPositionXEcefMeters" not in gnss_df.columns:
            return pd.DataFrame()

        # Filter valid rows: Must have valid WLS, Time, and measurements
        valid_mask = (
            gnss_df["WlsPositionXEcefMeters"].notna()
            & gnss_df["SvPositionXEcefMeters"].notna()
            & gnss_df["RawPseudorangeMeters"].notna()
        )
        df = gnss_df[valid_mask].copy()

        # Standardize time
        if "UnixTimeMillis" not in df.columns:
            if "utcTimeMillis" in df.columns:
                df["UnixTimeMillis"] = df["utcTimeMillis"]
            else:
                return pd.DataFrame()

        # Fix: Ensure timestamp is int64 for robust merging with GT
        df["UnixTimeMillis"] = df["UnixTimeMillis"].astype(np.int64)

        # Calculate Corrected Pseudorange
        # CorrectedPr = RawPr + SatBias (approx)
        if "SvClockBiasMeters" in df.columns:
            df["CorrectedPr"] = df["RawPseudorangeMeters"] + df[
                "SvClockBiasMeters"
            ].fillna(0)
        else:
            df["CorrectedPr"] = df["RawPseudorangeMeters"]

        # Weights based on Signal Strength: 10^(Cn0/10)
        df["weight"] = 10 ** (df["Cn0DbHz"].fillna(20) / 10.0)

        # Group by Epoch
        epochs = df.groupby("UnixTimeMillis")

        features_list = []

        # Prepare GT lookup if available (for Train/Val)
        gt_lookup = {}
        if gt_df is not None:
            # GT is usually 1Hz. Create a map UnixTime -> (Lat, Lon, Alt)
            gt_df = gt_df.dropna(subset=["LatitudeDegrees", "LongitudeDegrees"])
            for _, row in gt_df.iterrows():
                gt_lookup[int(row["UnixTimeMillis"])] = (
                    row["LatitudeDegrees"],
                    row["LongitudeDegrees"],
                    row.get("AltitudeMeters", 0),
                )

        for t, group in epochs:
            # User WLS Position (First valid in group)
            wls_x = group["WlsPositionXEcefMeters"].iloc[0]
            wls_y = group["WlsPositionYEcefMeters"].iloc[0]
            wls_z = group["WlsPositionZEcefMeters"].iloc[0]

            user_pos_ecef = np.array([wls_x, wls_y, wls_z])

            # Satellite Positions
            sat_pos_ecef = group[
                [
                    "SvPositionXEcefMeters",
                    "SvPositionYEcefMeters",
                    "SvPositionZEcefMeters",
                ]
            ].values

            # 1. Compute LOS vectors in ECEF
            u_vec_ecef, geom_dist = self.compute_los_vectors(
                sat_pos_ecef, user_pos_ecef
            )

            # 2. Rotate to ENU
            # Approximate Lat/Lon from WLS ECEF for rotation matrix
            p = np.sqrt(wls_x**2 + wls_y**2)
            lon_rad = np.arctan2(wls_y, wls_x)
            lat_rad = np.arctan2(wls_z, p)
            wls_lat = np.degrees(lat_rad)
            wls_lon = np.degrees(lon_rad)

            R = get_enu_rotation_matrix(wls_lat, wls_lon)

            # Rotate LOS vectors: (R @ u^T)^T
            u_vec_enu = (R @ u_vec_ecef.T).T  # (N, 3)

            # 3. Pseudorange Residuals
            # Residual = CorrectedPr - GeometricDist
            # Center to remove Rx Clock Bias and common errors
            raw_res = group["CorrectedPr"].values - geom_dist
            if len(raw_res) > 0:
                pr_res_centered = raw_res - np.median(raw_res)
            else:
                pr_res_centered = raw_res

            # 4. Doppler Residuals
            # Rate = -V_sat . u + V_rx . u + drift
            # Expected Rate (Sat only) = - (V_sat . u)
            sat_vel = group[
                [
                    "SvVelocityXEcefMetersPerSecond",
                    "SvVelocityYEcefMetersPerSecond",
                    "SvVelocityZEcefMetersPerSecond",
                ]
            ].values
            v_sat_dot_u = np.sum(sat_vel * u_vec_ecef, axis=1)

            # Measured Rate
            meas_rate = group["PseudorangeRateMetersPerSecond"].values

            # Diff = Measured - Expected(SatMotion) ~= V_rx . u + drift
            # Note: Sign convention of PrRate varies. Assuming standard: PrRate = RangeRate.
            # If RangeRate = (V_sat - V_rx).u = V_sat.u - V_rx.u
            # Then Meas - V_sat.u = -V_rx.u
            # Centering removes drift and bulk user motion, leaving local inconsistencies

            if np.isnan(v_sat_dot_u).any():
                v_sat_dot_u = np.nan_to_num(v_sat_dot_u)

            dop_res_raw = meas_rate - v_sat_dot_u
            if len(dop_res_raw) > 0:
                dop_res_centered = dop_res_raw - np.median(dop_res_raw)
            else:
                dop_res_centered = dop_res_raw

            # 5. Compute Forces
            weights = group["weight"].values

            f_pr = self.compute_forces(pr_res_centered, u_vec_enu, weights)
            f_dop = self.compute_forces(dop_res_centered, u_vec_enu, weights)

            # 6. Geometry Stiffness (DOP-like)
            # Weighted covariance of U vectors
            if np.sum(weights) > 0:
                # Weighted outer product: (N, 3, 1) * (N, 1, 3) -> (N, 3, 3)
                u_outer = u_vec_enu[:, :, np.newaxis] * u_vec_enu[:, np.newaxis, :]
                G = np.sum(
                    weights[:, np.newaxis, np.newaxis] * u_outer, axis=0
                ) / np.sum(weights)
                g_xx, g_yy, g_zz = G[0, 0], G[1, 1], G[2, 2]
            else:
                g_xx, g_yy, g_zz = 0.0, 0.0, 0.0

            # 7. Aggregates
            cn0_mean = group["Cn0DbHz"].mean()
            sv_count = len(group)

            # Feature Row
            row_dict = {
                "UnixTimeMillis": t,
                "Wls_Lat": wls_lat,
                "Wls_Lon": wls_lon,
                "Wls_Alt": wls_z,  # Approx
                "WlsPositionXEcefMeters": wls_x,
                "WlsPositionYEcefMeters": wls_y,
                "WlsPositionZEcefMeters": wls_z,
                "F_pr_E": f_pr[0],
                "F_pr_N": f_pr[1],
                "F_pr_U": f_pr[2],
                "F_dop_E": f_dop[0],
                "F_dop_N": f_dop[1],
                "F_dop_U": f_dop[2],
                "G_xx": g_xx,
                "G_yy": g_yy,
                "G_zz": g_zz,
                "Cn0_mean": cn0_mean,
                "Sv_count": sv_count,
            }

            # 8. Target Calculation (Train/Val only)
            if gt_lookup and t in gt_lookup:
                gt_lat, gt_lon, gt_alt = gt_lookup[t]

                # Convert GT to ECEF
                gt_x, gt_y, gt_z = wgs84_to_ecef(gt_lat, gt_lon, gt_alt)

                # Vector WLS -> GT in ECEF
                d_ecef_x = gt_x - wls_x
                d_ecef_y = gt_y - wls_y
                d_ecef_z = gt_z - wls_z

                # Rotate to ENU (centered at WLS)
                d_enu = R @ np.array([d_ecef_x, d_ecef_y, d_ecef_z])

                row_dict["Target_E"] = d_enu[0]
                row_dict["Target_N"] = d_enu[1]
                row_dict["Target_U"] = d_enu[2]

            features_list.append(row_dict)

        result_df = pd.DataFrame(features_list)

        # Save to cache
        try:
            result_df.to_parquet(cache_path, index=False)
        except Exception:
            pass

        return result_df

    def create_features(self, split="train", load_cached_data=True):
        """
        Main pipeline to create features for a dataset split.
        """
        # Check global cache first
        if split == "train":
            global_cache = CACHE_TRAIN_FEATURES
        elif split == "val":
            global_cache = CACHE_VAL_FEATURES
        else:
            global_cache = CACHE_TEST_FEATURES

        if load_cached_data and os.path.exists(global_cache):
            print(f"Loading {split} features from {global_cache}...")
            return pd.read_parquet(global_cache)

        print(f"Generating {split} features...")
        meta_df = self.loader.load_metadata(split)

        # Get unique drives
        unique_drives = meta_df[["drive_id", "phone_name"]].drop_duplicates()

        all_features = []

        for _, row in tqdm(
            unique_drives.iterrows(),
            total=len(unique_drives),
            desc=f"Processing {split}",
        ):
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]

            drive_feats = self.process_drive(
                drive_id, phone_name, split, load_cached_data
            )

            if not drive_feats.empty:
                # Add tripId for identification
                trip_id = f"{drive_id}-{phone_name}"
                drive_feats["tripId"] = trip_id
                all_features.append(drive_feats)

        if not all_features:
            print("No features generated!")
            return pd.DataFrame()

        final_df = pd.concat(all_features, ignore_index=True)

        # Save global cache
        try:
            final_df.to_parquet(global_cache, index=False)
        except Exception as e:
            print(f"Failed to save global cache: {e}")

        return final_df
