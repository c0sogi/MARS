import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from library.config import Config
from library.utils import WGS84Utils


class GNSSPreprocessor:
    def __init__(self):
        self.wgs84 = WGS84Utils()

    def ecef_to_lla(self, x, y, z):
        """
        Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.
        """
        # WGS84 ellipsoid constants
        a = 6378137.0
        e = 8.1819190842622e-2

        b = np.sqrt(a**2 * (1 - e**2))
        ep = np.sqrt((a**2 - b**2) / b**2)

        p = np.sqrt(x**2 + y**2)
        th = np.arctan2(a * z, b * p)

        lon = np.arctan2(y, x)
        lat = np.arctan2(
            (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
        )

        # Convert to degrees
        lon = np.degrees(lon)
        lat = np.degrees(lat)

        return lat, lon

    def load_raw_data(self, gnss_path):
        """
        Load raw GNSS data from CSV.
        """
        full_path = os.path.join(Config.INPUT_DIR, gnss_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"GNSS file not found: {full_path}")

        # Only read necessary columns to save memory/time
        # We need WLS positions to compute baseline
        use_cols = [
            "utcTimeMillis",
            "Cn0DbHz",
            "SvElevationDegrees",
            "SvAzimuthDegrees",
            "SignalType",
            "AccumulatedDeltaRangeState",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        # Read CSV
        # Note: Some files might have missing columns, handle gracefully if needed,
        # but competition data structure is consistent.
        df = pd.read_csv(full_path)

        # Filter valid signals (basic cleaning)
        # Drop rows where essential measurements are missing
        df = df.dropna(subset=["Cn0DbHz", "SvElevationDegrees"])

        return df

    def stratify_satellites(self, df):
        """
        Partition satellites into three strata based on signal quality and type.
        """
        # Stratum 1: Global (All visible satellites)
        s1 = df.copy()

        # Stratum 2: High-Precision
        # Criteria: SignalType in {L5, E5a, B2a, J5} OR Valid Carrier Phase
        # Valid Carrier Phase is usually bit 0 of AccumulatedDeltaRangeState
        high_prec_signals = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]

        # Check ADR Valid (Bit 0 is 1)
        adr_state = df["AccumulatedDeltaRangeState"].fillna(0).astype(int)
        adr_valid = (adr_state & 1) == 1

        sig_valid = df["SignalType"].isin(high_prec_signals)

        s2 = df[sig_valid | adr_valid].copy()

        # Stratum 3: High-Risk
        # Criteria: Elevation < 30 degrees
        s3 = df[df["SvElevationDegrees"] < 30].copy()

        return s1, s2, s3

    def aggregate_features(self, df_stratum, timestamp_col="UnixTimeMillis"):
        """
        Compute statistics (mean, std, min, max) for features within a stratum.
        """
        if df_stratum.empty:
            return pd.DataFrame()

        # Group by timestamp
        agg_funcs = {
            "Cn0DbHz": ["mean", "std", "min", "max"],
            "SvElevationDegrees": ["mean", "std", "min", "max"],
        }

        grouped = df_stratum.groupby(timestamp_col).agg(agg_funcs)

        # Flatten MultiIndex columns
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

        return grouped

    def compute_azimuth_context(self, df, timestamp_col="UnixTimeMillis"):
        """
        Compute signal-weighted azimuth sine and cosine.
        """
        if df.empty:
            return pd.DataFrame(columns=["Azimuth_Sin", "Azimuth_Cos"])

        df = df.copy()
        rads = np.radians(df["SvAzimuthDegrees"])
        # Weight by linear power (10^(dB/10))
        weights = np.power(10, df["Cn0DbHz"] / 10.0)

        df["sin_az"] = np.sin(rads) * weights
        df["cos_az"] = np.cos(rads) * weights
        df["weight"] = weights

        grouped = df.groupby(timestamp_col)[["sin_az", "cos_az", "weight"]].sum()

        # Normalize
        # Add epsilon to avoid division by zero
        grouped["Azimuth_Sin"] = grouped["sin_az"] / (grouped["weight"] + 1e-9)
        grouped["Azimuth_Cos"] = grouped["cos_az"] / (grouped["weight"] + 1e-9)

        return grouped[["Azimuth_Sin", "Azimuth_Cos"]]

    def process_drive(self, drive_id, phone_name, gnss_path, gt_df=None):
        """
        Process a single drive/phone sequence.
        """
        # 1. Load Data
        raw_df = self.load_raw_data(gnss_path)

        # 2. Temporal Quantization
        # Round utcTimeMillis to nearest second (1000 ms)
        raw_df["UnixTimeMillis"] = np.round(raw_df["utcTimeMillis"] / 1000.0) * 1000
        raw_df["UnixTimeMillis"] = raw_df["UnixTimeMillis"].astype(np.int64)

        # 3. Extract WLS Baseline
        # Take the first available WLS position per timestamp
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        # Group by time and take first non-null
        wls_df = raw_df.groupby("UnixTimeMillis")[wls_cols].first().reset_index()

        # Convert WLS ECEF to Lat/Lon
        wls_lat, wls_lon = self.ecef_to_lla(
            wls_df["WlsPositionXEcefMeters"].values,
            wls_df["WlsPositionYEcefMeters"].values,
            wls_df["WlsPositionZEcefMeters"].values,
        )
        wls_df["WLS_Lat"] = wls_lat
        wls_df["WLS_Lon"] = wls_lon

        # 4. Stratification
        s1, s2, s3 = self.stratify_satellites(raw_df)

        # 5. Aggregation
        feat_s1 = self.aggregate_features(s1, "UnixTimeMillis")
        feat_s2 = self.aggregate_features(s2, "UnixTimeMillis")
        feat_s3 = self.aggregate_features(s3, "UnixTimeMillis")

        # Rename columns
        feat_s1.columns = [f"S1_{c}" for c in feat_s1.columns]
        feat_s2.columns = [f"S2_{c}" for c in feat_s2.columns]
        feat_s3.columns = [f"S3_{c}" for c in feat_s3.columns]

        # 6. Context Features (from Global Stratum S1)
        context = self.compute_azimuth_context(s1, "UnixTimeMillis")

        # 7. Merge Features
        # Start with WLS baseline timestamps
        merged = wls_df.set_index("UnixTimeMillis")
        merged = merged.join([feat_s1, feat_s2, feat_s3, context], how="left")

        # Fill NaNs (e.g. empty strata) with 0
        merged = merged.fillna(0)

        # 8. Add Targets (if GT provided)
        if gt_df is not None:
            # Align GT timestamps
            gt_df = gt_df.copy()
            gt_df["UnixTimeMillis"] = np.round(gt_df["UnixTimeMillis"] / 1000.0) * 1000
            gt_df["UnixTimeMillis"] = gt_df["UnixTimeMillis"].astype(np.int64)

            # Inner join to keep only timestamps where we have both GNSS and GT
            merged = merged.join(
                gt_df.set_index("UnixTimeMillis")[
                    ["LatitudeDegrees", "LongitudeDegrees"]
                ],
                how="inner",
            )

            # Compute Residuals (Meters)
            d_north, d_east = self.wgs84.degrees_to_meters(
                merged["LatitudeDegrees"].values,
                merged["LongitudeDegrees"].values,
                merged["WLS_Lat"].values,
                merged["WLS_Lon"].values,
            )

            merged["Target_North"] = d_north
            merged["Target_East"] = d_east

            # Drop absolute GT coordinates
            merged = merged.drop(columns=["LatitudeDegrees", "LongitudeDegrees"])

        # Reset index
        merged = merged.reset_index()

        # Add identifiers
        merged["drive_id"] = drive_id
        merged["phone_name"] = phone_name

        return merged

    def _process_dataset(self, meta_path, cache_path, load_cached_data, is_test=False):
        # Check cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Processing data from {meta_path}...")
        meta_df = pd.read_csv(meta_path)

        results = []

        # Identify unique trips
        trips = (
            meta_df.groupby(["drive_id", "phone_name", "gnss_path"])
            .first()
            .reset_index()
        )

        for _, row in tqdm(
            trips.iterrows(), total=len(trips), desc="Processing Drives"
        ):
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]
            gnss_path = row["gnss_path"]

            gt_subset = None
            if not is_test:
                # Get GT for this trip
                gt_subset = meta_df[
                    (meta_df["drive_id"] == drive_id)
                    & (meta_df["phone_name"] == phone_name)
                ][["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

            try:
                processed_df = self.process_drive(
                    drive_id, phone_name, gnss_path, gt_subset
                )

                if is_test:
                    # Filter to requested timestamps
                    # Note: Sample submission timestamps might not be exactly 1000ms aligned in raw file,
                    # but our rounding logic should align them if they are close.
                    # We rely on the rounded UnixTimeMillis matching.
                    req_timestamps = meta_df[
                        (meta_df["drive_id"] == drive_id)
                        & (meta_df["phone_name"] == phone_name)
                    ]["UnixTimeMillis"].values

                    # Round requested timestamps too just in case
                    req_timestamps = np.round(req_timestamps / 1000.0) * 1000
                    req_timestamps = req_timestamps.astype(np.int64)

                    processed_df = processed_df[
                        processed_df["UnixTimeMillis"].isin(req_timestamps)
                    ]

                results.append(processed_df)

            except Exception as e:
                print(f"Error processing {drive_id} {phone_name}: {e}")
                continue

        if not results:
            raise ValueError("No data processed!")

        final_df = pd.concat(results, ignore_index=True)

        # Sort
        final_df = final_df.sort_values(["drive_id", "phone_name", "UnixTimeMillis"])

        # Save cache
        print(f"Saving processed data to {cache_path}")
        final_df.to_parquet(cache_path, index=False)

        return final_df

    def process_train_data(self, load_cached_data=True):
        return self._process_dataset(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_CACHE_PATH,
            load_cached_data,
            is_test=False,
        )

    def process_val_data(self, load_cached_data=True):
        return self._process_dataset(
            Config.VAL_METADATA_PATH,
            Config.VAL_CACHE_PATH,
            load_cached_data,
            is_test=False,
        )

    def process_test_data(self, load_cached_data=True):
        return self._process_dataset(
            Config.TEST_METADATA_PATH,
            Config.TEST_CACHE_PATH,
            load_cached_data,
            is_test=True,
        )
