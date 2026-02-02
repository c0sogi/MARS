import pandas as pd
import numpy as np
import os
from library.utils import CoordinateTransformer, IOHelper


class FeatureEngineer:
    def __init__(self, config=None):
        self.config = config or {}
        self.cache_dir = "./working/idea_20/"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Signal Bands
        self.L1_BANDS = ["GPS_L1", "GAL_E1", "GLO_G1", "QZS_J1", "BDS_B1I", "BDS_B1C"]
        self.L5_BANDS = ["GPS_L5", "GAL_E5A", "QZS_J5", "BDS_B2A"]

    def _calculate_residuals(self, df):
        """
        Calculate pseudorange and doppler residuals.
        """
        # 1. Calculate Geometric Range (Sat to WLS)
        dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
        dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
        dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Unit vectors (LOS)
        df["ux"] = dx / dist
        df["uy"] = dy / dist
        df["uz"] = dz / dist

        # 2. Corrected Pseudorange
        # CorrectedPr = RawPr + SatClkBias - Isrb - Iono - Tropo
        # Note: RawPseudorangeMeters includes Receiver Clock Bias.
        # The residual will be (RecClkBias + PositionError_proj).
        # We fill NaNs with 0 for corrections to avoid dropping rows, assuming small impact or handled by robust model
        df["CorrectedPr"] = (
            df["RawPseudorangeMeters"]
            + df["SvClockBiasMeters"].fillna(0)
            - df["IsrbMeters"].fillna(0)
            - df["IonosphericDelayMeters"].fillna(0)
            - df["TroposphericDelayMeters"].fillna(0)
        )

        # Pseudorange Residual
        df["pr_residual"] = df["CorrectedPr"] - dist

        # 3. Doppler Residual
        # Rate = u * (v_sat - v_rec) + clk_drift
        # We assume v_rec = 0 for the "force" calculation (static assumption residual)
        # or we just take the projection of satellite velocity.
        # Doppler Residual = Rate - u * v_sat
        vx = df["SvVelocityXEcefMetersPerSecond"].fillna(0)
        vy = df["SvVelocityYEcefMetersPerSecond"].fillna(0)
        vz = df["SvVelocityZEcefMetersPerSecond"].fillna(0)

        sat_proj = df["ux"] * vx + df["uy"] * vy + df["uz"] * vz
        df["doppler_residual"] = df["PseudorangeRateMetersPerSecond"] - sat_proj

        # 4. Weights
        # Inverse variance weighting
        # Add epsilon to avoid div by zero
        df["weight"] = 1.0 / (df["RawPseudorangeUncertaintyMeters"] + 1e-6)

        return df

    def _compute_epoch_features(self, df_epoch):
        """
        Aggregates features for a single epoch (timestamp).
        """
        # Basic info
        res = {}

        # Rotation Matrix to ENU
        # Use WLS position of the first satellite entry (they are same for the epoch)
        wls_x = df_epoch["WlsPositionXEcefMeters"].iloc[0]
        wls_y = df_epoch["WlsPositionYEcefMeters"].iloc[0]
        wls_z = df_epoch["WlsPositionZEcefMeters"].iloc[0]

        lat, lon, _ = CoordinateTransformer.ecef_to_wgs84(wls_x, wls_y, wls_z)
        sin_lat = np.sin(np.radians(lat))
        cos_lat = np.cos(np.radians(lat))
        sin_lon = np.sin(np.radians(lon))
        cos_lon = np.cos(np.radians(lon))

        def rotate_to_enu(fx, fy, fz):
            e = -sin_lon * fx + cos_lon * fy
            n = -sin_lat * cos_lon * fx - sin_lat * sin_lon * fy + cos_lat * fz
            u = cos_lat * cos_lon * fx + cos_lat * sin_lon * fy + sin_lat * fz
            return e, n, u

        # Groups
        l1_mask = df_epoch["SignalType"].isin(self.L1_BANDS)
        l5_mask = df_epoch["SignalType"].isin(self.L5_BANDS)

        # Helper to compute force
        def compute_force(sub_df, residual_col):
            if sub_df.empty:
                return 0.0, 0.0, 0.0, 0.0

            w = sub_df["weight"]
            r = sub_df[residual_col]
            ux, uy, uz = sub_df["ux"], sub_df["uy"], sub_df["uz"]

            # Weighted sum of residual vectors
            # Force = Sum( w * r * u ) / Sum(w)
            # Normalizing by sum of weights to make it intensive (like an average error vector)
            sum_w = w.sum()
            fx = (w * r * ux).sum() / sum_w
            fy = (w * r * uy).sum() / sum_w
            fz = (w * r * uz).sum() / sum_w

            return fx, fy, fz, sum_w

        # L1 Force
        fx_l1, fy_l1, fz_l1, w_l1 = compute_force(df_epoch[l1_mask], "pr_residual")
        e_l1, n_l1, u_l1 = rotate_to_enu(fx_l1, fy_l1, fz_l1)
        res["L1_force_e"] = e_l1
        res["L1_force_n"] = n_l1
        res["L1_force_u"] = u_l1
        res["L1_weight_sum"] = w_l1
        res["L1_count"] = l1_mask.sum()

        # L5 Force
        fx_l5, fy_l5, fz_l5, w_l5 = compute_force(df_epoch[l5_mask], "pr_residual")
        e_l5, n_l5, u_l5 = rotate_to_enu(fx_l5, fy_l5, fz_l5)
        res["L5_force_e"] = e_l5
        res["L5_force_n"] = n_l5
        res["L5_force_u"] = u_l5
        res["L5_weight_sum"] = w_l5
        res["L5_count"] = l5_mask.sum()

        # Doppler Force (All bands)
        fx_dop, fy_dop, fz_dop, w_dop = compute_force(df_epoch, "doppler_residual")
        e_dop, n_dop, u_dop = rotate_to_enu(fx_dop, fy_dop, fz_dop)
        res["Dop_force_e"] = e_dop
        res["Dop_force_n"] = n_dop
        res["Dop_force_u"] = u_dop

        # Signal Quality
        res["CN0_mean"] = df_epoch["Cn0DbHz"].mean()
        res["CN0_max"] = df_epoch["Cn0DbHz"].max()

        # IMU features (if available in columns)
        # Assuming they are already merged and constant for the epoch
        imu_cols = [c for c in df_epoch.columns if "Uncal" in c or "Measurement" in c]
        for c in imu_cols:
            res[c] = df_epoch[c].iloc[0]

        return pd.Series(res)

    def process_trip(self, df_gnss, load_cached_data=True):
        """
        Process a single trip dataframe to generate features.
        """
        if df_gnss.empty:
            return pd.DataFrame()

        trip_id = df_gnss["tripId"].iloc[0]
        cache_file = f"features_{trip_id}.parquet"

        # Check cache
        if load_cached_data:
            cached_df = IOHelper.load_parquet(cache_file)
            if cached_df is not None:
                return cached_df

        # Calculate Residuals
        df_gnss = self._calculate_residuals(df_gnss)

        # Group by Epoch and Compute Features
        # We use apply. For large datasets, iterating or vectorized operations are faster,
        # but apply is flexible for this complex logic.
        features = df_gnss.groupby("utcTimeMillis").apply(self._compute_epoch_features)

        # Reset index to get utcTimeMillis back
        features = features.reset_index()
        features["tripId"] = trip_id

        # Save to cache
        IOHelper.save_parquet(features, cache_file)

        return features

    def create_features(self, trips_metadata, load_cached_data=True):
        """
        Main entry point to create features for a list of trips.
        trips_metadata: DataFrame containing tripId and file paths.
        """
        # We need to load the raw data using the paths in metadata
        # However, the GnssDataset loader loads everything into one big DF which might be OOM.
        # Better to iterate trips, load raw, process, then clear.

        # We'll use a local instance of GnssDataset helper logic or just direct read since we have paths.
        # But GnssDataset._process_trip handles IMU merging which is useful.
        # We can't import GnssDataset here easily without circular dep or re-import.
        # We will assume the user passes a list of dataframes or we re-implement the load loop using utils.

        # Actually, the requirement says "Import the functions or classes from the given Python files".
        # So we can import GnssDataset.
        from library.data_loader import GnssDataset

        loader = GnssDataset(root_dir="./input")

        all_features = []

        unique_trips = trips_metadata[
            ["tripId", "gnss_path", "imu_path"]
        ].drop_duplicates()

        for _, row in unique_trips.iterrows():
            trip_id = row["tripId"]

            # Check if feature cache exists to skip loading raw data
            cache_file = f"features_{trip_id}.parquet"
            if load_cached_data and os.path.exists(
                os.path.join(self.cache_dir, cache_file)
            ):
                print(f"Loading features for {trip_id} from cache...")
                feat_df = IOHelper.load_parquet(cache_file)
                all_features.append(feat_df)
                continue

            # Load raw data
            print(f"Processing features for {trip_id}...")
            try:
                # We use the internal method of GnssDataset to load specific trip
                # Note: We don't need GT here for feature generation, just GNSS/IMU
                df_raw = loader._process_trip(
                    trip_id, row["gnss_path"], row["imu_path"], gt_rel_path=None
                )

                if not df_raw.empty:
                    feat_df = self.process_trip(
                        df_raw, load_cached_data=False
                    )  # Already checked cache
                    all_features.append(feat_df)
            except Exception as e:
                print(f"Error processing {trip_id}: {e}")

        if not all_features:
            return pd.DataFrame()

        return pd.concat(all_features, ignore_index=True)
