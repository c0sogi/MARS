import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import WGS84


class PreProcessor:
    def __init__(self):
        self.wgs84 = WGS84()
        self.feature_cols = []  # To be populated after processing

    def _compute_features(self, gnss_df):
        """
        Aggregates raw GNSS data into Sky Panorama and Global features at 1Hz.
        """
        # 1. Temporal Quantization (Round to nearest second)
        # utcTimeMillis is the raw timestamp. We round to align with GT.
        gnss_df["UnixTimeMillis"] = np.round(gnss_df["utcTimeMillis"] / 1000.0) * 1000.0
        gnss_df["UnixTimeMillis"] = gnss_df["UnixTimeMillis"].astype(np.int64)

        # 2. Filter invalid measurements
        # We need Azimuth, Elevation, and Cn0 for our features
        valid_mask = (
            gnss_df["SvAzimuthDegrees"].notna()
            & gnss_df["SvElevationDegrees"].notna()
            & gnss_df["Cn0DbHz"].notna()
        )
        df = gnss_df[valid_mask].copy()

        if df.empty:
            return pd.DataFrame()

        # 3. Azimuthal Binning
        # Bin index = floor(Azimuth / (360 / bins)) % bins
        bin_width = 360.0 / Config.NUM_AZIMUTH_BINS
        df["azimuth_bin"] = (df["SvAzimuthDegrees"] // bin_width).astype(
            int
        ) % Config.NUM_AZIMUTH_BINS

        # 4. Aggregations
        # We aggregate by Timestamp.

        # A. Global Stats
        global_aggs = df.groupby("UnixTimeMillis").agg(
            {
                "Cn0DbHz": ["mean", "std", "min", "max"],
                "SvElevationDegrees": ["mean", "std", "min", "max"],
                "Svid": "count",
                "RawPseudorangeUncertaintyMeters": "mean",
            }
        )
        global_aggs.columns = [
            "global_cn0_mean",
            "global_cn0_std",
            "global_cn0_min",
            "global_cn0_max",
            "global_elev_mean",
            "global_elev_std",
            "global_elev_min",
            "global_elev_max",
            "global_sat_count",
            "global_pr_unc_mean",
        ]

        # Fill NaN std with 0 (happens if only 1 satellite)
        global_aggs.fillna(0, inplace=True)

        # B. Sky Panorama (Per Bin Stats)
        # We need to pivot: index=Timestamp, columns=Bin, values=Stats
        # First, aggregate per timestamp + bin
        bin_aggs = (
            df.groupby(["UnixTimeMillis", "azimuth_bin"])
            .agg({"Cn0DbHz": "max", "SvElevationDegrees": "mean", "Svid": "count"})
            .reset_index()
        )

        # Pivot to wide format
        panorama = bin_aggs.pivot(
            index="UnixTimeMillis",
            columns="azimuth_bin",
            values=["Cn0DbHz", "SvElevationDegrees", "Svid"],
        )

        # Flatten columns: e.g., Cn0DbHz_0, Cn0DbHz_1, ...
        # Note: The pivot creates a MultiIndex columns.
        new_cols = []
        for metric, bin_idx in panorama.columns:
            if metric == "Cn0DbHz":
                name = f"bin_{bin_idx}_cn0_max"
            elif metric == "SvElevationDegrees":
                name = f"bin_{bin_idx}_elev_mean"
            else:  # Svid
                name = f"bin_{bin_idx}_sat_count"
            new_cols.append(name)

        panorama.columns = new_cols

        # Ensure all bins exist (0 to NUM_BINS-1)
        expected_cols = []
        for i in range(Config.NUM_AZIMUTH_BINS):
            expected_cols.extend(
                [f"bin_{i}_cn0_max", f"bin_{i}_elev_mean", f"bin_{i}_sat_count"]
            )

        # Reindex to ensure all columns are present, fill missing with 0
        panorama = panorama.reindex(columns=expected_cols, fill_value=0)

        # 5. Merge Global and Panorama
        features = global_aggs.join(panorama, how="inner")

        # 6. Extract WLS Baseline Position (One per epoch)
        # WLS positions are repeated for all sats in an epoch, take the first one.
        # We handle NaNs in WLS columns by dropping them or filling?
        # Usually WLS is present. If not, we can't compute offset targets easily.
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        wls_pos = gnss_df.groupby("UnixTimeMillis")[wls_cols].first()

        features = features.join(wls_pos, how="inner")

        return features.reset_index()

    def _process_drive(self, drive_id, phone_name, gnss_path, gt_df=None):
        """
        Loads GNSS data for a drive, computes features, and calculates targets if GT is provided.
        """
        full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)
        if not os.path.exists(full_gnss_path):
            print(f"Warning: GNSS file not found: {full_gnss_path}")
            return pd.DataFrame()

        try:
            gnss_df = pd.read_csv(full_gnss_path)
        except Exception as e:
            print(f"Error reading {full_gnss_path}: {e}")
            return pd.DataFrame()

        # Compute Features
        processed_df = self._compute_features(gnss_df)

        if processed_df.empty:
            return pd.DataFrame()

        # Add Metadata
        processed_df["drive_id"] = drive_id
        processed_df["phone_name"] = phone_name

        # If Ground Truth is provided (Train/Val), compute targets
        if gt_df is not None:
            # Filter GT for this drive/phone just in case, though passed df should be specific
            # GT timestamps are already in UnixTimeMillis

            # Merge features with GT
            merged_df = pd.merge(processed_df, gt_df, on="UnixTimeMillis", how="inner")

            if merged_df.empty:
                return pd.DataFrame()

            # Calculate Targets (ENU Offsets)
            # 1. Get WLS ECEF (from features)
            wls_x = merged_df["WlsPositionXEcefMeters"].values
            wls_y = merged_df["WlsPositionYEcefMeters"].values
            wls_z = merged_df["WlsPositionZEcefMeters"].values

            # 2. Convert WLS ECEF to LLA (Reference for ENU)
            ref_lat, ref_lon, ref_alt = self.wgs84.ecef_to_geodetic(wls_x, wls_y, wls_z)

            # Store WLS Lat/Lon for reconstruction verification or analysis
            merged_df["wls_lat"] = ref_lat
            merged_df["wls_lon"] = ref_lon

            # 3. Get GT LLA
            gt_lat = merged_df["LatitudeDegrees"].values
            gt_lon = merged_df["LongitudeDegrees"].values
            gt_alt = merged_df["AltitudeMeters"].values  # GT usually has altitude

            # 4. Convert GT LLA to ECEF
            gt_x, gt_y, gt_z = self.wgs84.geodetic_to_ecef(gt_lat, gt_lon, gt_alt)

            # 5. Calculate Difference in ECEF
            dx = gt_x - wls_x
            dy = gt_y - wls_y
            dz = gt_z - wls_z

            # 6. Convert Difference to ENU (Target)
            # Note: We use the WLS position as the reference point for the local tangent plane
            # Formulas:
            # t = cos(lat) * cos(lon) * dx + cos(lat) * sin(lon) * dy + sin(lat) * dz
            # e = -sin(lon) * dx + cos(lon) * dy
            # n = -sin(lat) * cos(lon) * dx - sin(lat) * sin(lon) * dy + cos(lat) * dz

            # Using vectorized numpy operations
            sin_lat = np.sin(np.radians(ref_lat))
            cos_lat = np.cos(np.radians(ref_lat))
            sin_lon = np.sin(np.radians(ref_lon))
            cos_lon = np.cos(np.radians(ref_lon))

            target_e = -sin_lon * dx + cos_lon * dy
            target_n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

            merged_df["target_east"] = target_e
            merged_df["target_north"] = target_n

            return merged_df

        else:
            # Test Mode: We still need WLS Lat/Lon for final reconstruction
            # Convert WLS ECEF to LLA
            wls_x = processed_df["WlsPositionXEcefMeters"].values
            wls_y = processed_df["WlsPositionYEcefMeters"].values
            wls_z = processed_df["WlsPositionZEcefMeters"].values

            ref_lat, ref_lon, ref_alt = self.wgs84.ecef_to_geodetic(wls_x, wls_y, wls_z)

            processed_df["wls_lat"] = ref_lat
            processed_df["wls_lon"] = ref_lon

            return processed_df

    def process_data(self, load_cached_data=True):
        """
        Main entry point to prepare Train, Val, and Test datasets.
        Handles caching logic.
        """
        train_cache = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
        val_cache = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
        test_cache = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

        # Check if cache exists
        cache_exists = (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        )

        if load_cached_data and cache_exists:
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df

        print("Processing data from scratch...")

        # Load Metadata
        meta_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        meta_val = pd.read_csv(Config.VAL_METADATA_PATH)
        meta_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # Helper to process a metadata dataframe
        def process_split(meta_df, is_test=False):
            results = []
            # Group by drive and phone to process each trip
            trips = meta_df.groupby(["drive_id", "phone_name"])

            for (drive_id, phone_name), group in trips:
                gnss_path = group.iloc[0]["gnss_path"]

                if is_test:
                    gt_df = None
                else:
                    # For train/val, we pass the GT subset
                    gt_df = group[
                        [
                            "UnixTimeMillis",
                            "LatitudeDegrees",
                            "LongitudeDegrees",
                            "AltitudeMeters",
                        ]
                    ].copy()

                processed_trip = self._process_drive(
                    drive_id, phone_name, gnss_path, gt_df
                )

                if not processed_trip.empty:
                    # For test set, we must filter to only the requested timestamps
                    if is_test:
                        requested_timestamps = group["UnixTimeMillis"].unique()
                        processed_trip = processed_trip[
                            processed_trip["UnixTimeMillis"].isin(requested_timestamps)
                        ]

                    results.append(processed_trip)

            if not results:
                return pd.DataFrame()
            return pd.concat(results, ignore_index=True)

        # Process Splits
        print("Processing Train...")
        train_df = process_split(meta_train, is_test=False)
        print("Processing Val...")
        val_df = process_split(meta_val, is_test=False)
        print("Processing Test...")
        test_df = process_split(meta_test, is_test=True)

        # Save to Cache
        print("Saving to cache...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

        return train_df, val_df, test_df
