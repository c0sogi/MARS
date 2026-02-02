import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import WGS84


class GNSSPreprocessor:
    """
    Handles loading, stratification, feature engineering, and target generation
    for GNSS data.
    """

    def __init__(self):
        self.config = Config()

    def _stratify_signals(self, df_gnss):
        """
        Creates boolean masks for signal stratification.

        Args:
            df_gnss: DataFrame containing raw GNSS measurements.

        Returns:
            df_gnss with added boolean columns: 'is_high_precision', 'is_high_risk'.
        """
        # High Precision: L5/E5a/B2a/J5 bands OR Valid Carrier Phase
        # Check SignalType for bands
        # SignalType might be null or string. Handle safely.

        # Create mask for bands
        # We need to escape special regex characters if any, but these are alphanumeric
        bands_pattern = "|".join(self.config.HIGH_PRECISION_BANDS)
        is_high_band = (
            df_gnss["SignalType"]
            .astype(str)
            .str.contains(bands_pattern, case=False, na=False)
        )

        # Check AccumulatedDeltaRangeState for Valid Carrier Phase (Bit 0)
        # We assume the column exists and is numeric.
        adr_state = df_gnss["AccumulatedDeltaRangeState"].fillna(0).astype(int)
        is_phase_valid = (adr_state & self.config.ADR_STATE_VALID_BIT) != 0

        df_gnss["is_high_precision"] = is_high_band | is_phase_valid

        # High Risk: Low Elevation
        df_gnss["is_high_risk"] = (
            df_gnss["SvElevationDegrees"] < self.config.ELEVATION_HIGH_RISK_THRESHOLD
        )

        return df_gnss

    def _compute_geometric_features(self, group):
        """
        Computes Signal-Weighted Azimuth Centroid features for a group of satellites (epoch).

        Args:
            group: DataFrame group for a specific timestamp.

        Returns:
            Series with AzimuthCentroidSin and AzimuthCentroidCos.
        """
        # Weights: Cn0DbHz (linear scale not strictly necessary for this heuristic, but commonly used directly)
        # Fill NaNs in Cn0 and Azimuth
        cn0 = group["Cn0DbHz"].fillna(0).values
        az_rad = np.radians(group["SvAzimuthDegrees"].fillna(0).values)

        if len(cn0) == 0 or np.sum(cn0) == 0:
            return pd.Series({"AzimuthCentroidSin": 0.0, "AzimuthCentroidCos": 0.0})

        # Weighted vector sum
        x = np.sum(cn0 * np.cos(az_rad))
        y = np.sum(cn0 * np.sin(az_rad))

        # Angle of resultant vector
        angle = np.arctan2(y, x)

        return pd.Series(
            {"AzimuthCentroidSin": np.sin(angle), "AzimuthCentroidCos": np.cos(angle)}
        )

    def _aggregate_features(self, df_gnss):
        """
        Aggregates raw GNSS data into stratified features per timestamp.

        Args:
            df_gnss: Preprocessed GNSS DataFrame with stratification flags.

        Returns:
            DataFrame with one row per timestamp and aggregated features.
        """
        # Define aggregations
        # We process each stratum separately and then merge

        # 1. Global Stratum (All satellites)
        global_agg = df_gnss.groupby("UnixTimeMillis")[self.config.RAW_FEATURES].agg(
            self.config.AGGREGATION_STATS
        )
        global_agg.columns = [f"Global_{c[0]}_{c[1]}" for c in global_agg.columns]

        # 2. High Precision Stratum
        hp_subset = df_gnss[df_gnss["is_high_precision"]]
        if not hp_subset.empty:
            hp_agg = hp_subset.groupby("UnixTimeMillis")[self.config.RAW_FEATURES].agg(
                self.config.AGGREGATION_STATS
            )
            hp_agg.columns = [f"HighPrecision_{c[0]}_{c[1]}" for c in hp_agg.columns]
        else:
            # Create empty DataFrame with correct index if subset is empty
            hp_agg = pd.DataFrame(index=global_agg.index)
            for feat in self.config.RAW_FEATURES:
                for stat in self.config.AGGREGATION_STATS:
                    hp_agg[f"HighPrecision_{feat}_{stat}"] = 0.0

        # 3. High Risk Stratum
        hr_subset = df_gnss[df_gnss["is_high_risk"]]
        if not hr_subset.empty:
            hr_agg = hr_subset.groupby("UnixTimeMillis")[self.config.RAW_FEATURES].agg(
                self.config.AGGREGATION_STATS
            )
            hr_agg.columns = [f"HighRisk_{c[0]}_{c[1]}" for c in hr_agg.columns]
        else:
            hr_agg = pd.DataFrame(index=global_agg.index)
            for feat in self.config.RAW_FEATURES:
                for stat in self.config.AGGREGATION_STATS:
                    hr_agg[f"HighRisk_{feat}_{stat}"] = 0.0

        # 4. Geometric Features (Global context)
        geom_agg = df_gnss.groupby("UnixTimeMillis").apply(
            self._compute_geometric_features
        )

        # Merge all
        df_features = pd.concat([global_agg, hp_agg, hr_agg, geom_agg], axis=1)

        # Fill NaNs resulting from empty strata for specific timestamps
        df_features = df_features.fillna(0.0)

        return df_features.reset_index()

    def _process_single_drive(self, drive_id, phone_name, gnss_rel_path, df_gt=None):
        """
        Processes a single drive: loads data, computes features, aligns with GT/WLS.

        Args:
            drive_id: Drive identifier.
            phone_name: Phone model name.
            gnss_rel_path: Relative path to device_gnss.csv.
            df_gt: DataFrame containing ground truth (optional, for train/val).

        Returns:
            DataFrame containing features and (optionally) targets for the drive.
        """
        gnss_path = os.path.join(self.config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found: {gnss_path}")
            return pd.DataFrame()

        # Load Raw GNSS
        # We only need specific columns to save memory
        use_cols = [
            "utcTimeMillis",
            "SignalType",
            "AccumulatedDeltaRangeState",
            "SvElevationDegrees",
            "SvAzimuthDegrees",
            "Cn0DbHz",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        try:
            df_gnss = pd.read_csv(gnss_path, usecols=lambda c: c in use_cols)
        except ValueError:
            # Fallback if some columns are missing (though they should exist based on description)
            df_gnss = pd.read_csv(gnss_path)

        # Time Alignment
        # Convert utcTimeMillis to UnixTimeMillis (rounded to nearest second)
        # Note: Input description says UnixTimeMillis is available in ground_truth,
        # but device_gnss has utcTimeMillis. They are essentially the same time scale.
        # We round to align with the 1Hz ground truth.
        df_gnss["UnixTimeMillis"] = np.round(df_gnss["utcTimeMillis"] / 1000.0) * 1000.0
        df_gnss["UnixTimeMillis"] = df_gnss["UnixTimeMillis"].astype(np.int64)

        # Extract WLS Baseline positions (one per epoch)
        # We take the first valid WLS position for each timestamp
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        if all(c in df_gnss.columns for c in wls_cols):
            df_wls = df_gnss.groupby("UnixTimeMillis")[wls_cols].first().reset_index()

            # Convert WLS ECEF to Geodetic
            lats, lons, alts = WGS84.ecef_to_geodetic(
                df_wls["WlsPositionXEcefMeters"].values,
                df_wls["WlsPositionYEcefMeters"].values,
                df_wls["WlsPositionZEcefMeters"].values,
            )
            df_wls["WlsLatitudeDegrees"] = lats
            df_wls["WlsLongitudeDegrees"] = lons
        else:
            # Should not happen based on dataset description, but handle gracefully
            return pd.DataFrame()

        # Stratify and Aggregate Features
        df_gnss = self._stratify_signals(df_gnss)
        df_features = self._aggregate_features(df_gnss)

        # Merge Features with WLS Baseline
        df_merged = pd.merge(
            df_features,
            df_wls[["UnixTimeMillis", "WlsLatitudeDegrees", "WlsLongitudeDegrees"]],
            on="UnixTimeMillis",
            how="inner",
        )

        # Add Metadata
        df_merged["drive_id"] = drive_id
        df_merged["phone_name"] = phone_name

        # Handle Ground Truth and Targets
        if df_gt is not None:
            # Ensure GT timestamps are aligned
            # GT is already 1Hz aligned in the metadata generation step usually,
            # but we ensure type consistency.
            df_gt = df_gt.copy()
            df_gt["UnixTimeMillis"] = df_gt["UnixTimeMillis"].astype(np.int64)

            # Merge GT
            # use inner join to keep only timestamps where we have both GNSS and GT
            df_final = pd.merge(
                df_merged,
                df_gt[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
                on="UnixTimeMillis",
                how="inner",
            )

            # Compute Regression Targets (Meters offset from WLS)
            # Target = GT - WLS
            d_north, d_east = WGS84.latlon_to_meters(
                df_final["LatitudeDegrees"] - df_final["WlsLatitudeDegrees"],
                df_final["LongitudeDegrees"] - df_final["WlsLongitudeDegrees"],
                df_final["WlsLatitudeDegrees"],  # Reference latitude
            )

            df_final["DeltaNorthMeters"] = d_north
            df_final["DeltaEastMeters"] = d_east

            return df_final
        else:
            # Test mode: No GT, no targets
            return df_merged

    def generate_dataset(self, split="train", load_cached_data=True):
        """
        Generates the processed dataset for a specific split.

        Args:
            split: 'train', 'val', or 'test'.
            load_cached_data: If True, attempts to load from parquet cache.

        Returns:
            Processed DataFrame.
        """
        cache_path = os.path.join(self.config.CACHE_DIR, f"{split}_processed.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Processing {split} data from raw files...")

        # 2. Load Metadata
        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
        else:
            meta_path = self.config.TEST_METADATA_PATH

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # 3. Process each drive
        # Group metadata by drive and phone to process sequentially
        grouped = df_meta.groupby(["drive_id", "phone_name"])

        results = []

        for (drive_id, phone_name), group in grouped:
            # For test set, group might have multiple rows but we just need the path from one
            # For train/val, group contains the GT rows

            gnss_path = group.iloc[0]["gnss_path"]

            if split in ["train", "val"]:
                df_drive = self._process_single_drive(
                    drive_id, phone_name, gnss_path, df_gt=group
                )
            else:
                df_drive = self._process_single_drive(
                    drive_id, phone_name, gnss_path, df_gt=None
                )
                # Filter to keep only requested timestamps in test
                requested_timestamps = group["UnixTimeMillis"].unique()
                df_drive = df_drive[
                    df_drive["UnixTimeMillis"].isin(requested_timestamps)
                ].copy()

                # Ensure we have rows for all requested timestamps?
                # If GNSS is missing for a timestamp, we might lose it.
                # For this competition, usually we predict for what we have or interpolate.
                # Here we return what we computed.

            if not df_drive.empty:
                results.append(df_drive)

        if not results:
            print(f"Warning: No data processed for split {split}")
            return pd.DataFrame()

        final_df = pd.concat(results, ignore_index=True)

        # 4. Save Cache
        print(f"Saving {split} data to cache: {cache_path}")
        final_df.to_parquet(cache_path, index=False)

        return final_df
