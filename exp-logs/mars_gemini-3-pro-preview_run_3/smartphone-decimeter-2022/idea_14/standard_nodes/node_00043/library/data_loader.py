import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from library.config import INPUT_DIR, METADATA_DIR, CACHE_DIR, FEATURES, SEED
from library.utils import ecef_to_geodetic, geodetic_to_ecef


class DataLoader:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()

    def load_metadata(self, split):
        """
        Load metadata for a specific split (train, val, test).
        """
        path = os.path.join(METADATA_DIR, f"{split}_metadata.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path)

    def _process_trip(self, trip_data):
        """
        Process a single trip: load GNSS/IMU, compute geometry-projected features, and align with GT.
        """
        trip_id = trip_data["tripId"]
        gnss_rel_path = trip_data["gnss_path"]
        imu_rel_path = trip_data["imu_path"]

        gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)
        imu_path = os.path.join(INPUT_DIR, imu_rel_path)

        # 1. Load GNSS Data
        gnss_cols = [
            "utcTimeMillis",
            "Svid",
            "Cn0DbHz",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "RawPseudorangeMeters",
            "SvClockBiasMeters",
            "IsrbMeters",
            "IonosphericDelayMeters",
            "TroposphericDelayMeters",
        ]

        try:
            # Read only necessary columns to save memory
            gnss_df = pd.read_csv(gnss_path, usecols=lambda c: c in gnss_cols)
        except Exception as e:
            print(f"Error reading GNSS for {trip_id}: {e}")
            return None

        # 2. Load IMU Data
        imu_cols = [
            "utcTimeMillis",
            "MessageType",
            "MeasurementX",
            "MeasurementY",
            "MeasurementZ",
        ]
        try:
            imu_df = pd.read_csv(imu_path, usecols=lambda c: c in imu_cols)
        except Exception:
            imu_df = pd.DataFrame(columns=imu_cols)

        # 3. Preprocess IMU (Aggregate dynamics)
        imu_feats = pd.DataFrame()
        if not imu_df.empty:
            accel = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
            if not accel.empty:
                accel["Accel_Mag"] = np.sqrt(
                    accel["MeasurementX"] ** 2
                    + accel["MeasurementY"] ** 2
                    + accel["MeasurementZ"] ** 2
                )
                # Align to nearest second
                accel["UnixTimeMillis"] = np.round(accel["utcTimeMillis"] / 1000) * 1000
                imu_agg = (
                    accel.groupby("UnixTimeMillis")["Accel_Mag"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                imu_agg.columns = ["UnixTimeMillis", "Accel_mean", "Accel_std"]
                imu_feats = imu_agg

        # 4. Process GNSS Features (GPR-Boost Logic)
        # Filter invalid rows
        gnss_df = gnss_df.dropna(
            subset=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
                "SvPositionXEcefMeters",
                "SvPositionYEcefMeters",
                "SvPositionZEcefMeters",
                "RawPseudorangeMeters",
            ]
        )

        if gnss_df.empty:
            return None

        gnss_df = gnss_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

        # Calculate Geometric Distance (Range)
        gnss_df["Range"] = np.sqrt(
            (gnss_df["SvPositionXEcefMeters"] - gnss_df["WlsPositionXEcefMeters"]) ** 2
            + (gnss_df["SvPositionYEcefMeters"] - gnss_df["WlsPositionYEcefMeters"])
            ** 2
            + (gnss_df["SvPositionZEcefMeters"] - gnss_df["WlsPositionZEcefMeters"])
            ** 2
        )

        # Calculate Corrected Pseudorange
        # Pr_corr = RawPr + SatClkBias - Isrb - Iono - Tropo
        gnss_df["Pr_corr"] = (
            gnss_df["RawPseudorangeMeters"]
            + gnss_df["SvClockBiasMeters"].fillna(0)
            - gnss_df["IsrbMeters"].fillna(0)
            - gnss_df["IonosphericDelayMeters"].fillna(0)
            - gnss_df["TroposphericDelayMeters"].fillna(0)
        )

        # Calculate Residual (contains Receiver Clock Bias + Position Error)
        gnss_df["Residual"] = gnss_df["Pr_corr"] - gnss_df["Range"]

        # Remove Common Mode Error (Receiver Clock Bias estimate)
        epoch_bias = (
            gnss_df.groupby("UnixTimeMillis")["Residual"].median().reset_index()
        )
        epoch_bias.columns = ["UnixTimeMillis", "ClockBias"]
        gnss_df = pd.merge(gnss_df, epoch_bias, on="UnixTimeMillis", how="left")
        gnss_df["DiffResidual"] = gnss_df["Residual"] - gnss_df["ClockBias"]

        # --- Geometric Projection ---
        # Get WLS Geodetic Coords for ENU rotation matrix
        wls_pos = gnss_df[
            [
                "UnixTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].drop_duplicates("UnixTimeMillis")

        # Convert WLS ECEF to Geodetic (Lat/Lon needed for rotation)
        def get_geo(row):
            lat, lon, alt = ecef_to_geodetic(
                row["WlsPositionXEcefMeters"],
                row["WlsPositionYEcefMeters"],
                row["WlsPositionZEcefMeters"],
            )
            return pd.Series({"Wls_Lat": lat, "Wls_Lon": lon})

        wls_geo = wls_pos.apply(get_geo, axis=1)
        wls_pos = pd.concat([wls_pos, wls_geo], axis=1)

        gnss_df = pd.merge(
            gnss_df,
            wls_pos[["UnixTimeMillis", "Wls_Lat", "Wls_Lon"]],
            on="UnixTimeMillis",
            how="left",
        )

        # Calculate Line-of-Sight (LOS) Unit Vectors in ECEF
        dx = gnss_df["SvPositionXEcefMeters"] - gnss_df["WlsPositionXEcefMeters"]
        dy = gnss_df["SvPositionYEcefMeters"] - gnss_df["WlsPositionYEcefMeters"]
        dz = gnss_df["SvPositionZEcefMeters"] - gnss_df["WlsPositionZEcefMeters"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        ux_ecef = dx / dist
        uy_ecef = dy / dist
        uz_ecef = dz / dist

        # Rotate LOS vectors to local ENU frame
        lat_rad = np.radians(gnss_df["Wls_Lat"].values)
        lon_rad = np.radians(gnss_df["Wls_Lon"].values)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Rotation matrix application (ECEF -> ENU)
        u_e = -sin_lon * ux_ecef + cos_lon * uy_ecef
        u_n = (
            -sin_lat * cos_lon * ux_ecef
            - sin_lat * sin_lon * uy_ecef
            + cos_lat * uz_ecef
        )

        # Weighting by Signal Strength
        w = gnss_df["Cn0DbHz"].fillna(20)

        # Compute Projected Residual Forces
        gnss_df["Force_E"] = w * gnss_df["DiffResidual"] * u_e
        gnss_df["Force_N"] = w * gnss_df["DiffResidual"] * u_n

        # Compute Geometry Covariance terms (Weighted)
        gnss_df["Cov_E"] = w * (u_e**2)
        gnss_df["Cov_N"] = w * (u_n**2)
        gnss_df["Cov_EN"] = w * (u_e * u_n)

        # Aggregate per Epoch
        agg_funcs = {
            "Force_E": "sum",
            "Force_N": "sum",
            "Cov_E": "sum",
            "Cov_N": "sum",
            "Cov_EN": "sum",
            "Cn0DbHz": "mean",
            "Svid": "count",
            "Wls_Lat": "first",
            "Wls_Lon": "first",
            "WlsPositionXEcefMeters": "first",
            "WlsPositionYEcefMeters": "first",
            "WlsPositionZEcefMeters": "first",
        }

        epoch_feats = gnss_df.groupby("UnixTimeMillis").agg(agg_funcs).reset_index()

        # Rename to match config features
        epoch_feats = epoch_feats.rename(
            columns={
                "Force_E": "NetForce_E",
                "Force_N": "NetForce_N",
                "Cn0DbHz": "Cn0DbHz_mean",
                "Svid": "Svid_count",
            }
        )

        # 5. Merge IMU Features
        if not imu_feats.empty:
            epoch_feats = pd.merge(
                epoch_feats, imu_feats, on="UnixTimeMillis", how="left"
            )
            epoch_feats["Accel_mean"] = epoch_feats["Accel_mean"].fillna(9.8)
            epoch_feats["Accel_std"] = epoch_feats["Accel_std"].fillna(0)
        else:
            epoch_feats["Accel_mean"] = 9.8
            epoch_feats["Accel_std"] = 0

        # 6. Merge Ground Truth (if available) and Compute Targets
        if "gt_path" in trip_data and pd.notna(trip_data["gt_path"]):
            gt_path = os.path.join(INPUT_DIR, trip_data["gt_path"])
            if os.path.exists(gt_path):
                gt_df = pd.read_csv(gt_path)
                gt_df = gt_df[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

                # Inner join to keep only labeled epochs
                merged_df = pd.merge(
                    epoch_feats, gt_df, on="UnixTimeMillis", how="inner"
                )

                # Compute Targets: ENU difference between GT and WLS
                # 1. Convert GT Lat/Lon to ECEF (assume Alt=0 or WLS Alt, horizontal error is insensitive to small alt errors)
                # We use WLS altitude to minimize vertical error projection into horizontal
                # But to be precise, we convert WLS ECEF to Geodetic to get WLS Alt first (done above)
                # Actually, we can just use 0 for both if we only care about relative horizontal offset roughly,
                # but better to use WLS Alt for GT conversion to keep them on same shell.

                # Re-calculate WLS ECEF from WLS Lat/Lon/Alt to ensure consistency?
                # We have WLS ECEF in columns.
                # Let's convert GT Lat/Lon + WLS Alt to ECEF
                # Note: We need WLS Alt. We didn't save it in agg.
                # Let's re-derive WLS Alt from ECEF columns
                _, _, wls_alt = ecef_to_geodetic(
                    merged_df["WlsPositionXEcefMeters"].values,
                    merged_df["WlsPositionYEcefMeters"].values,
                    merged_df["WlsPositionZEcefMeters"].values,
                )

                gt_x, gt_y, gt_z = geodetic_to_ecef(
                    merged_df["LatitudeDegrees"].values,
                    merged_df["LongitudeDegrees"].values,
                    wls_alt,
                )

                # Calculate ENU difference
                dx = gt_x - merged_df["WlsPositionXEcefMeters"].values
                dy = gt_y - merged_df["WlsPositionYEcefMeters"].values
                dz = gt_z - merged_df["WlsPositionZEcefMeters"].values

                # Rotate difference to ENU using WLS Lat/Lon
                lat_rad = np.radians(merged_df["Wls_Lat"].values)
                lon_rad = np.radians(merged_df["Wls_Lon"].values)
                sin_lat = np.sin(lat_rad)
                cos_lat = np.cos(lat_rad)
                sin_lon = np.sin(lon_rad)
                cos_lon = np.cos(lon_rad)

                target_e = -sin_lon * dx + cos_lon * dy
                target_n = (
                    -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
                )

                merged_df["target_E"] = target_e
                merged_df["target_N"] = target_n

                merged_df["tripId"] = trip_id
                merged_df["drive_id"] = trip_data["drive_id"]

                return merged_df

        # Test set (no GT)
        epoch_feats["tripId"] = trip_id
        epoch_feats["drive_id"] = trip_data["drive_id"]
        return epoch_feats

    def load_dataset(self, split, load_cached_data=True):
        """
        Load and process the dataset for the given split.
        """
        cache_file = os.path.join(CACHE_DIR, f"{split}_features.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split} data from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Processing {split} data from scratch...")
        meta = self.load_metadata(split)

        # Get unique trips to process
        unique_trips = meta[
            ["tripId", "drive_id", "phone_name", "gnss_path", "imu_path"]
        ].drop_duplicates()

        if "gt_path" in meta.columns:
            unique_trips["gt_path"] = meta.groupby("tripId")["gt_path"].transform(
                "first"
            )

        trip_list = [row for _, row in unique_trips.iterrows()]

        results = []
        # Process trips in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for res in executor.map(self._process_trip, trip_list):
                if res is not None:
                    results.append(res)

        if not results and split != "test":
            raise ValueError("No data processed!")

        if results:
            final_df = pd.concat(results, ignore_index=True)
        else:
            final_df = pd.DataFrame()

        keys = ["tripId", "UnixTimeMillis"]

        if split == "test":
            # Cite debug_lesson_4: Preserve Row Cardinality During Inference
            # For test set, ensure all rows from metadata are present (Left Join on meta)
            final_df = pd.merge(meta[keys], final_df, on=keys, how="left")

            # Sort for interpolation
            final_df = final_df.sort_values(by=["tripId", "UnixTimeMillis"])

            # Interpolate WLS Baseline columns
            wls_cols = [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
                "Wls_Lat",
                "Wls_Lon",
            ]

            # Define interpolation function (Linear + Fill edges)
            def interpolate_group(group):
                if group.notna().sum() > 1:
                    return (
                        group.interpolate(method="linear", limit_direction="both")
                        .ffill()
                        .bfill()
                    )
                else:
                    return group.ffill().bfill()  # Fallback

            # Apply interpolation per trip
            for col in wls_cols:
                if col not in final_df.columns:
                    final_df[col] = np.nan

                # Use transform for efficiency
                final_df[col] = final_df.groupby("tripId")[col].transform(
                    interpolate_group
                )

            # Fill remaining NaNs in WLS with 0 (Last resort)
            final_df[wls_cols] = final_df[wls_cols].fillna(0)

            # Fill Feature columns with 0 (assuming zero residual)
            for feat in FEATURES:
                if feat not in final_df.columns:
                    final_df[feat] = 0.0
                else:
                    final_df[feat] = final_df[feat].fillna(0.0)

        else:
            # Train/Val: Inner join to keep only valid data
            final_df = pd.merge(final_df, meta[keys], on=keys, how="inner")

        print(f"Saving {split} data to cache: {cache_file}")
        final_df.to_parquet(cache_file, index=False)

        return final_df


def get_train_data(load_cached_data=True):
    """
    Loads the training dataset using the DataLoader.
    Ensures caching logic is followed.
    """
    loader = DataLoader()
    return loader.load_dataset("train", load_cached_data=load_cached_data)


def get_val_data(load_cached_data=True):
    """
    Loads the validation dataset using the DataLoader.
    Ensures caching logic is followed.
    """
    loader = DataLoader()
    return loader.load_dataset("val", load_cached_data=load_cached_data)


def get_test_data(load_cached_data=True):
    """
    Loads the test dataset using the DataLoader.
    Ensures caching logic is followed.
    """
    loader = DataLoader()
    return loader.load_dataset("test", load_cached_data=load_cached_data)
