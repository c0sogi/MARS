import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.utils import WGS84Converter


class GNSSPreprocessor:
    """
    Handles raw GNSS data loading, cleaning, feature engineering, and target generation.
    """

    def __init__(self):
        self.converter = WGS84Converter()
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def ecef_to_lla(self, x, y, z):
        """
        Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.
        Using WGS84 ellipsoid constants from Config.
        """
        a = Config.WGS84_A
        f = Config.WGS84_F
        b = a * (1 - f)
        e2 = 2 * f - f**2
        ep2 = (a**2 - b**2) / b**2

        r = np.sqrt(x**2 + y**2)
        E2 = a**2 - b**2
        F = 54 * b**2 * z**2
        G = r**2 + (1 - e2) * z**2 - e2 * E2
        C = (e2**2 * F * r**2) / (G**3)
        S = (1 + C + np.sqrt(C**2 + 2 * C)) ** (1 / 3)
        P = F / (3 * (S + 1 / S + 1) ** 2 * G**2)
        Q = np.sqrt(1 + 2 * e2**2 * P)
        ro = -(P * e2 * r) / (1 + Q) + np.sqrt(
            (a**2 / 2) * (1 + 1 / Q)
            - (P * (1 - e2) * z**2) / (Q * (1 + Q))
            - P * r**2 / 2
        )
        U = np.sqrt((r - e2 * ro) ** 2 + z**2)
        V = np.sqrt((r - e2 * ro) ** 2 + (1 - e2) * z**2)
        zo = (b**2 * z) / (a * V)

        height = U * (1 - b**2 / (a * V))
        lat = np.arctan((z + ep2 * zo) / r)
        lon = np.arctan2(y, x)

        return np.degrees(lat), np.degrees(lon), height

    def _align_timestamps(self, df):
        """
        Aligns raw GNSS logs to 1Hz by rounding utcTimeMillis.
        Renames utcTimeMillis to UnixTimeMillis for consistency with Ground Truth.
        """
        # utcTimeMillis is in ms. Round to nearest second (1000ms)
        df["UnixTimeMillis"] = (np.round(df["utcTimeMillis"] / 1000) * 1000).astype(
            np.int64
        )
        return df

    def _aggregate_global_features(self, df_grouped):
        """
        Computes global statistics for signal quality across all satellites per epoch.
        """
        # Aggregation dictionary mapping columns to list of functions
        agg_dict = {
            "Cn0DbHz": ["mean", "std", "min", "max"],
            "SvElevationDegrees": ["mean", "std", "min", "max"],
            "RawPseudorangeUncertaintyMeters": ["mean"],
            "Svid": ["count"],
        }

        df_agg = df_grouped.agg(agg_dict)

        # Flatten MultiIndex columns
        df_agg.columns = [
            f"{col}_{stat}" if col != "Svid" else "SatCount"
            for col, stat in df_agg.columns
        ]

        # Rename RawPseudorangeUncertaintyMeters_mean to match Config
        df_agg = df_agg.rename(
            columns={
                "RawPseudorangeUncertaintyMeters_mean": "RawPseudorangeUncertaintyMeters_mean"
            }
        )

        return df_agg

    def _aggregate_directional_features(self, df):
        """
        Computes statistics per azimuthal quadrant (NE, SE, SW, NW).
        """
        # Define quadrants
        # NE: 0-90, SE: 90-180, SW: 180-270, NW: 270-360
        df["quadrant"] = pd.cut(
            df["SvAzimuthDegrees"],
            bins=[0, 90, 180, 270, 360],
            labels=["NE", "SE", "SW", "NW"],
            include_lowest=True,
        )

        # Group by Time and Quadrant
        quad_agg = (
            df.groupby(["UnixTimeMillis", "quadrant"])
            .agg({"Cn0DbHz": "mean", "Svid": "count"})
            .unstack(fill_value=0)
        )

        # Flatten columns: e.g., (Cn0DbHz, NE) -> NE_Cn0DbHz_mean
        new_cols = []
        for metric, quad in quad_agg.columns:
            if metric == "Cn0DbHz":
                new_cols.append(f"{quad}_{metric}_mean")
            elif metric == "Svid":
                new_cols.append(f"{quad}_SatCount")

        quad_agg.columns = new_cols
        return quad_agg

    def _process_drive(self, drive_id, phone_name, gnss_path, gt_df=None):
        """
        Processes a single drive: loads GNSS, computes WLS baseline, aggregates features,
        and merges with ground truth (if provided).
        """
        full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)
        if not os.path.exists(full_gnss_path):
            print(f"Warning: GNSS file not found: {full_gnss_path}")
            return None

        # Load Raw GNSS
        # Only load necessary columns to save memory
        use_cols = [
            "utcTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "SvElevationDegrees",
            "SvAzimuthDegrees",
            "Cn0DbHz",
            "RawPseudorangeUncertaintyMeters",
            "Svid",
        ]
        df_gnss = pd.read_csv(full_gnss_path, usecols=lambda c: c in use_cols)

        # Align Timestamps
        df_gnss = self._align_timestamps(df_gnss)

        # Extract Baseline WLS Position (Take first valid per epoch)
        # Note: WLS position is repeated for all sats in an epoch
        df_wls = df_gnss.groupby("UnixTimeMillis")[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].first()

        # Convert WLS ECEF to LLA
        wls_lat, wls_lon, _ = self.ecef_to_lla(
            df_wls["WlsPositionXEcefMeters"].values,
            df_wls["WlsPositionYEcefMeters"].values,
            df_wls["WlsPositionZEcefMeters"].values,
        )
        df_wls["wls_lat"] = wls_lat
        df_wls["wls_lon"] = wls_lon

        # Feature Engineering
        grouped = df_gnss.groupby("UnixTimeMillis")

        # 1. Global Features
        df_global = self._aggregate_global_features(grouped)

        # 2. Directional Features
        df_directional = self._aggregate_directional_features(df_gnss)

        # Merge Features and Baseline
        df_features = df_global.join(df_directional).join(
            df_wls[["wls_lat", "wls_lon"]]
        )

        # Add Metadata
        df_features["drive_id"] = drive_id
        df_features["phone_name"] = phone_name
        df_features = df_features.reset_index()  # UnixTimeMillis becomes a column

        # Merge with Ground Truth if available (Train/Val)
        if gt_df is not None:
            # Cite debug_lesson_4: Normalize Timestamp Precision Before Merging Time-Series Data
            # GT timestamps are precise (e.g., ...432), while GNSS features are rounded (e.g., ...000).
            # We must round GT timestamps to match the feature grid for the merge to succeed.
            gt_df = gt_df.copy()
            gt_df["UnixTimeMillis"] = (
                np.round(gt_df["UnixTimeMillis"] / 1000) * 1000
            ).astype(np.int64)

            # GT usually has UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
            df_merged = pd.merge(df_features, gt_df, on="UnixTimeMillis", how="inner")

            # Compute Targets (Delta Meters)
            d_east, d_north = self.converter.deg_to_meters(
                df_merged["LatitudeDegrees"].values,
                df_merged["LongitudeDegrees"].values,
                df_merged["wls_lat"].values,
                df_merged["wls_lon"].values,
            )
            df_merged["d_east"] = d_east
            df_merged["d_north"] = d_north

            return df_merged
        else:
            # Test set - no GT, no targets
            return df_features

    def process_data(
        self, metadata_path, dataset_name, load_cached_data=True, debug=False
    ):
        """
        Main entry point to process a dataset (train, val, or test).
        Handles caching and iteration over metadata.
        """
        cache_file = os.path.join(self.cache_dir, f"{dataset_name}_processed.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {dataset_name} data from {cache_file}...")
            return pd.read_parquet(cache_file)

        print(f"Processing {dataset_name} data from scratch...")
        df_meta = pd.read_csv(metadata_path)

        if debug:
            print(
                f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows from metadata."
            )
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        # Unique drive-phone pairs
        pairs = df_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

        processed_dfs = []

        for _, row in tqdm(
            pairs.iterrows(), total=len(pairs), desc=f"Processing {dataset_name}"
        ):
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]
            gnss_path = row["gnss_path"]

            # Filter GT for this pair if it exists in metadata
            gt_subset = None
            if "LatitudeDegrees" in df_meta.columns:
                gt_subset = df_meta[
                    (df_meta["drive_id"] == drive_id)
                    & (df_meta["phone_name"] == phone_name)
                ][["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

            df_proc = self._process_drive(drive_id, phone_name, gnss_path, gt_subset)

            if df_proc is not None and not df_proc.empty:
                # If this is test data, we need to ensure we only keep rows requested in sample_submission
                # The metadata for test is derived from sample_submission, so we filter by timestamps in meta
                if "LatitudeDegrees" not in df_meta.columns:
                    req_timestamps = df_meta[
                        (df_meta["drive_id"] == drive_id)
                        & (df_meta["phone_name"] == phone_name)
                    ]["UnixTimeMillis"].values

                    # Cite debug_lesson_4: Normalize Timestamp Precision
                    # The features in df_proc have rounded timestamps (e.g., ...000).
                    # The requested timestamps are exact (e.g., ...432).
                    # We create a mapping to filter the features AND restore exact timestamps for submission.
                    mapping_df = pd.DataFrame({"ExactMillis": req_timestamps})
                    mapping_df["RoundedMillis"] = (
                        np.round(mapping_df["ExactMillis"] / 1000) * 1000
                    ).astype(np.int64)

                    # Merge to filter and map
                    df_proc = pd.merge(
                        df_proc,
                        mapping_df,
                        left_on="UnixTimeMillis",
                        right_on="RoundedMillis",
                        how="inner",
                    )

                    # Swap back to exact timestamps for the dataset
                    df_proc = df_proc.drop(columns=["UnixTimeMillis", "RoundedMillis"])
                    df_proc = df_proc.rename(columns={"ExactMillis": "UnixTimeMillis"})

                processed_dfs.append(df_proc)

        if not processed_dfs:
            raise ValueError(
                f"No data processed for {dataset_name}. Check paths and metadata."
            )

        final_df = pd.concat(processed_dfs, ignore_index=True)

        # Fill missing features (e.g. if a quadrant had no satellites)
        final_df = final_df.fillna(0)

        # Save to cache
        print(f"Saving {dataset_name} data to {cache_file}...")
        final_df.to_parquet(cache_file, index=False)

        return final_df

    def get_train_data(self, load_cached_data=True, debug=False):
        return self.process_data(
            Config.TRAIN_METADATA_PATH, "train", load_cached_data, debug
        )

    def get_val_data(self, load_cached_data=True, debug=False):
        return self.process_data(
            Config.VAL_METADATA_PATH, "val", load_cached_data, debug
        )

    def get_test_data(self, load_cached_data=True, debug=False):
        return self.process_data(
            Config.TEST_METADATA_PATH, "test", load_cached_data, debug
        )
