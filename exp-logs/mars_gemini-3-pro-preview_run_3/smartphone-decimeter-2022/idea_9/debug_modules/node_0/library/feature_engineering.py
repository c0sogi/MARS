import os
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import DataLoader
from library.doppler_processing import DopplerVelocityEstimator
from library.utils import GeoUtils


class FeatureGenerator:
    """
    Orchestrates the creation of model input features for the Doppler-Aided Residual Boosting pipeline.
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.doppler_estimator = DopplerVelocityEstimator()
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_cache_path(self, split: str) -> str:
        return os.path.join(self.working_dir, f"{split}_features.parquet")

    def _process_imu(self, imu_df):
        """
        Aggregates IMU data to 1Hz resolution.
        """
        if imu_df.empty:
            return pd.DataFrame()

        # Filter for Accelerometer
        acc_df = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
        if acc_df.empty:
            return pd.DataFrame()

        # Calculate magnitude
        acc_df["acc_mag"] = np.sqrt(
            acc_df["MeasurementX"] ** 2
            + acc_df["MeasurementY"] ** 2
            + acc_df["MeasurementZ"] ** 2
        )

        # Round timestamp to nearest second to align with GNSS
        # utcTimeMillis is in ms.
        acc_df["UnixTimeMillis"] = (
            np.round(acc_df["utcTimeMillis"] / 1000) * 1000
        ).astype(np.int64)

        # Aggregate
        agg_funcs = {"acc_mag": ["mean", "std"]}
        imu_agg = acc_df.groupby(["tripId", "UnixTimeMillis"]).agg(agg_funcs)
        imu_agg.columns = [f"imu_{col[0]}_{col[1]}" for col in imu_agg.columns.values]
        imu_agg.reset_index(inplace=True)

        return imu_agg

    def _process_gnss_signal(self, gnss_df):
        """
        Aggregates GNSS signal quality metrics.
        """
        if gnss_df.empty:
            return pd.DataFrame()

        # Rename for consistency
        gnss_df = gnss_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

        # Aggregations
        agg_funcs = {
            "Cn0DbHz": ["mean", "max", "std"],
            "SvElevationDegrees": ["mean"],
            "Svid": ["count"],
        }

        gnss_agg = gnss_df.groupby(["tripId", "UnixTimeMillis"]).agg(agg_funcs)
        gnss_agg.columns = [
            f"signal_{col[0]}_{col[1]}" for col in gnss_agg.columns.values
        ]
        gnss_agg.reset_index(inplace=True)

        # Extract WLS Position (take first valid per epoch)
        # We need WLS position to compute targets and for inference baseline
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        wls_df = (
            gnss_df[["tripId", "UnixTimeMillis"] + wls_cols]
            .groupby(["tripId", "UnixTimeMillis"])
            .first()
            .reset_index()
        )

        # Convert WLS ECEF to LLA
        lats, lons, alts = GeoUtils.ecef_to_lla(
            wls_df["WlsPositionXEcefMeters"].values,
            wls_df["WlsPositionYEcefMeters"].values,
            wls_df["WlsPositionZEcefMeters"].values,
        )
        wls_df["WlsLat"] = lats
        wls_df["WlsLon"] = lons
        wls_df["WlsAlt"] = alts

        # Merge
        feat_df = pd.merge(
            gnss_agg, wls_df, on=["tripId", "UnixTimeMillis"], how="left"
        )

        return feat_df

    def _compute_wls_kinematics(self, df):
        """
        Computes baseline speed from WLS positions.
        Expects df sorted by tripId and Time.
        """
        # Ensure sorted
        df = df.sort_values(["tripId", "UnixTimeMillis"])

        # Calculate deltas
        dt = df.groupby("tripId")["UnixTimeMillis"].diff() / 1000.0

        # We need ECEF deltas to compute distance accurately
        # But we might only have LLA if we dropped ECEF.
        # Let's use Haversine on Lat/Lon for horizontal speed approx

        lat = df["WlsLat"].values
        lon = df["WlsLon"].values

        # Shifted arrays
        lat_prev = np.roll(lat, 1)
        lon_prev = np.roll(lon, 1)

        # Mask first elements of groups
        is_start = df["tripId"] != df["tripId"].shift(1)

        dists = GeoUtils.haversine_distance(lat_prev, lon_prev, lat, lon)
        dists[is_start] = np.nan

        speed = dists / dt
        df["wls_speed"] = speed

        # Fill NaNs (first points) with 0 or next valid
        df["wls_speed"] = df["wls_speed"].fillna(0)

        return df

    def _compute_targets(self, df, gt_df):
        """
        Computes ENU residuals between GT and WLS.
        Target = GT - WLS (in meters)
        """
        # Merge GT
        # GT has LatitudeDegrees, LongitudeDegrees
        df_merged = pd.merge(
            df,
            gt_df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            on=["tripId", "UnixTimeMillis"],
            how="inner",
            suffixes=("", "_gt"),
        )

        # Convert GT LLA to ECEF
        # Use WLS Alt for conversion to minimize Z-axis projection noise.
        gt_x, gt_y, gt_z = GeoUtils.lla_to_ecef(
            df_merged["LatitudeDegrees"].values,
            df_merged["LongitudeDegrees"].values,
            df_merged["WlsAlt"].values,
        )

        wls_x = df_merged["WlsPositionXEcefMeters"].values
        wls_y = df_merged["WlsPositionYEcefMeters"].values
        wls_z = df_merged["WlsPositionZEcefMeters"].values

        # WLS LLA for reference frame
        ref_lat = df_merged["WlsLat"].values
        ref_lon = df_merged["WlsLon"].values

        # Vectorized ECEF to ENU
        dx = gt_x - wls_x
        dy = gt_y - wls_y
        dz = gt_z - wls_z

        lat_rad = np.deg2rad(ref_lat)
        lon_rad = np.deg2rad(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

        df_merged["target_east"] = e
        df_merged["target_north"] = n

        return df_merged

    def generate_features(
        self, split: str, load_cached_data: bool = True, limit: int = None
    ):
        """
        Main pipeline to generate features for a specific split.
        """
        cache_path = self._get_cache_path(split)

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features for {split} from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {split} (Limit: {limit})...")

        # 2. Load Raw Data
        meta_df = self.data_loader.load_metadata(split)

        # Filter metadata if limit is set
        if limit:
            unique_drives = meta_df["drive_id"].unique()[:limit]
            meta_df = meta_df[meta_df["drive_id"].isin(unique_drives)]
            print(f"Limited to {len(unique_drives)} drives.")

        # Load GNSS
        gnss_df = self.data_loader.load_gnss(
            split, load_cached_data=load_cached_data, limit=limit
        )

        # Load IMU
        imu_df = self.data_loader.load_imu(
            split, load_cached_data=load_cached_data, limit=limit
        )

        # 3. Compute Features

        # A. Doppler Velocity
        print("Estimating Doppler velocity...")
        doppler_df = self.doppler_estimator.estimate_velocity(
            gnss_df, split, load_cached_data=load_cached_data
        )

        # B. GNSS Signal Features & WLS Position
        print("Processing GNSS signal features...")
        signal_df = self._process_gnss_signal(gnss_df)

        # C. IMU Features
        print("Processing IMU features...")
        imu_feat_df = self._process_imu(imu_df)

        # 4. Merge Features
        # Base is the metadata timestamps
        base_df = meta_df[["tripId", "UnixTimeMillis", "drive_id", "phone_name"]].copy()

        # Merge Doppler
        if not doppler_df.empty:
            base_df = pd.merge(
                base_df,
                doppler_df[["tripId", "UnixTimeMillis", "v_east", "v_north", "speed"]],
                on=["tripId", "UnixTimeMillis"],
                how="left",
            )

        # Merge Signal (includes WLS position)
        if not signal_df.empty:
            base_df = pd.merge(
                base_df, signal_df, on=["tripId", "UnixTimeMillis"], how="left"
            )

        # Merge IMU
        if not imu_feat_df.empty:
            base_df = pd.merge(
                base_df, imu_feat_df, on=["tripId", "UnixTimeMillis"], how="left"
            )

        # 5. Post-Merge Features
        print("Computing kinematics...")
        base_df = self._compute_wls_kinematics(base_df)

        # 6. Compute Targets (if Train/Val)
        if split in ["train", "val"]:
            print("Computing targets...")
            # Metadata already has GT Lat/Lon
            gt_cols = [
                "tripId",
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
            ]
            gt_subset = meta_df[gt_cols]

            base_df = self._compute_targets(base_df, gt_subset)

            # Filter out rows with NaN targets (if WLS was missing)
            base_df = base_df.dropna(subset=["target_east", "target_north"])

        # 7. Final Cleanup
        # Fill NaNs in velocity columns with 0
        vel_cols = ["v_east", "v_north", "speed"]
        for c in vel_cols:
            if c in base_df.columns:
                base_df[c] = base_df[c].fillna(0)

        # Drop ECEF columns to save space
        cols_to_drop = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        base_df.drop(
            columns=[c for c in cols_to_drop if c in base_df.columns], inplace=True
        )

        # 8. Save Cache
        print(f"Saving features to {cache_path}...")
        base_df.to_parquet(cache_path, index=False)

        return base_df
