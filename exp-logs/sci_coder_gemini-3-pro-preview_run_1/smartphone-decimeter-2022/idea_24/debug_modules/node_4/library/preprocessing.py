import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import WGS84Utils


class GnssPreprocessor:
    def __init__(self):
        self.config = Config

    def _filter_hq_signals(self, df):
        """
        Filters GNSS data for high-quality signals (Stream B).
        Criteria: SignalType in HQ list OR ADR State has Valid bit set.
        """
        # Check Signal Type
        cond_signal_type = df["SignalType"].isin(self.config.HQ_SIGNAL_TYPES)

        # Check ADR State (Bitwise AND)
        # Handle potential NaNs in AccumulatedDeltaRangeState by filling with 0
        adr_state = df["AccumulatedDeltaRangeState"].fillna(0).astype(int)
        cond_adr_valid = (adr_state & self.config.ADR_STATE_VALID_BIT) != 0

        return df[cond_signal_type | cond_adr_valid].copy()

    def _aggregate_stream(self, df, prefix):
        """
        Aggregates features for a stream (A or B) by epoch.
        """
        if df.empty:
            # Return empty dataframe with expected columns if input is empty
            cols = []
            for feat in self.config.STAT_FEATURES:
                for stat in self.config.STATS_LIST:
                    cols.append(f"{prefix}_{feat}_{stat}")
            for feat in self.config.MEAN_FEATURES:
                cols.append(f"{prefix}_{feat}_mean")
            cols.append(f"{prefix}_sat_count")
            # We need to return a DataFrame that can be joined, so it needs the index name if possible,
            # but since it's empty, we just return empty with columns.
            return pd.DataFrame(columns=cols)

        # Group by timestamp (rounded)
        grouped = df.groupby("UnixTimeMillis")

        agg_dict = {}

        # Stat features
        for feat in self.config.STAT_FEATURES:
            agg_dict[feat] = self.config.STATS_LIST

        # Mean features
        for feat in self.config.MEAN_FEATURES:
            agg_dict[feat] = ["mean"]

        # Count feature (using Svid)
        agg_dict[self.config.COUNT_FEATURE] = ["count"]

        # Perform aggregation
        agg_df = grouped.agg(agg_dict)

        # Flatten columns
        new_columns = []
        for col_name, stat_name in agg_df.columns:
            if col_name == self.config.COUNT_FEATURE and stat_name == "count":
                new_columns.append(f"{prefix}_sat_count")
            else:
                new_columns.append(f"{prefix}_{col_name}_{stat_name}")

        agg_df.columns = new_columns
        return agg_df

    def _process_drive(self, drive_id, phone_name, gnss_path, gt_path=None):
        """
        Process a single drive: load, clean, feature engineer, and merge targets.
        """
        full_gnss_path = os.path.join(self.config.INPUT_DIR, gnss_path)
        if not os.path.exists(full_gnss_path):
            print(f"Warning: GNSS file not found: {full_gnss_path}")
            return None

        # Load GNSS
        # We only need specific columns to save memory
        use_cols = (
            [
                "utcTimeMillis",
                "SignalType",
                "AccumulatedDeltaRangeState",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
            + self.config.STAT_FEATURES
            + self.config.MEAN_FEATURES
            + [self.config.COUNT_FEATURE]
        )

        try:
            # Attempt to read only necessary columns
            gnss_df = pd.read_csv(full_gnss_path, usecols=lambda c: c in use_cols)
        except ValueError:
            # Fallback if column names mismatch or other read issues
            gnss_df = pd.read_csv(full_gnss_path)

        # 1. Temporal Quantization
        # utcTimeMillis is roughly UnixTimeMillis. Round to nearest second.
        gnss_df["UnixTimeMillis"] = np.round(gnss_df["utcTimeMillis"] / 1000) * 1000
        gnss_df["UnixTimeMillis"] = gnss_df["UnixTimeMillis"].astype(np.int64)

        # 2. Extract WLS Baseline (One per epoch)
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        # Drop rows where WLS is NaN or 0.0 (invalid fix)
        # Cite debug_lesson_25: Avoid Indiscriminate Zero-Filling on Coordinate Reference Data
        wls_df = gnss_df[["UnixTimeMillis"] + wls_cols].copy()
        wls_df[wls_cols] = wls_df[wls_cols].replace(0.0, np.nan)
        wls_df = wls_df.dropna(subset=wls_cols)

        # Cite debug_lesson_30: Filter Coordinates by Physical Validity, Not Just Sentinel Zeros
        # Calculate geocentric radius to filter out points near Earth's center (invalid fixes)
        wls_radius = np.sqrt(
            wls_df["WlsPositionXEcefMeters"] ** 2
            + wls_df["WlsPositionYEcefMeters"] ** 2
            + wls_df["WlsPositionZEcefMeters"] ** 2
        )
        # Earth radius is approx 6,371 km. Filter out anything < 6,000 km.
        wls_df = wls_df[wls_radius > 6.0e6].copy()

        # Drop duplicates to get one position per epoch
        wls_df = wls_df.drop_duplicates(subset=["UnixTimeMillis"]).copy()

        if wls_df.empty:
            return None

        # Convert WLS ECEF to Lat/Lon/Alt
        x = wls_df["WlsPositionXEcefMeters"].values
        y = wls_df["WlsPositionYEcefMeters"].values
        z = wls_df["WlsPositionZEcefMeters"].values

        lat, lon, alt = WGS84Utils.ecef_to_geodetic(x, y, z)
        wls_df["wls_lat"] = lat
        wls_df["wls_lon"] = lon
        wls_df["wls_alt"] = alt

        # Keep only necessary WLS columns
        wls_df = wls_df[["UnixTimeMillis", "wls_lat", "wls_lon", "wls_alt"]]

        # 3. Feature Engineering
        # Stream A: All signals
        feat_a = self._aggregate_stream(gnss_df, prefix="A")

        # Stream B: High Quality
        hq_df = self._filter_hq_signals(gnss_df)
        feat_b = self._aggregate_stream(hq_df, prefix="B")

        # Merge Features (A join B)
        # Ensure we keep all epochs from A (which should cover all epochs in GNSS log)
        features = feat_a.join(feat_b, how="left")

        # Fill NaNs in Stream B (caused by epochs with 0 HQ satellites) with 0
        features = features.fillna(0)

        # Merge with WLS baseline
        features = features.merge(wls_df, on="UnixTimeMillis", how="inner")

        # Add Metadata
        features["drive_id"] = drive_id
        features["phone_name"] = phone_name

        # 4. Target Generation (if GT exists)
        if gt_path:
            full_gt_path = os.path.join(self.config.INPUT_DIR, gt_path)

            if os.path.exists(full_gt_path):
                gt_df = pd.read_csv(full_gt_path)
                gt_df["UnixTimeMillis"] = (
                    np.round(gt_df["UnixTimeMillis"] / 1000) * 1000
                )
                gt_df["UnixTimeMillis"] = gt_df["UnixTimeMillis"].astype(np.int64)

                # Merge features with GT
                merged = pd.merge(
                    features,
                    gt_df[
                        [
                            "UnixTimeMillis",
                            "LatitudeDegrees",
                            "LongitudeDegrees",
                            "AltitudeMeters",
                        ]
                    ],
                    on="UnixTimeMillis",
                    how="inner",
                )

                if merged.empty:
                    return None

                # Compute Cartesian Residuals (ENU)
                # Vector from WLS (Reference) to GT (Target)

                wls_lat = merged["wls_lat"].values
                wls_lon = merged["wls_lon"].values
                wls_alt = merged["wls_alt"].values

                gt_lat = merged["LatitudeDegrees"].values
                gt_lon = merged["LongitudeDegrees"].values
                gt_alt = merged["AltitudeMeters"].values

                # Convert GT to ECEF
                gt_x, gt_y, gt_z = WGS84Utils.geodetic_to_ecef(gt_lat, gt_lon, gt_alt)

                # Convert GT ECEF to ENU relative to WLS
                e, n, u = WGS84Utils.ecef_to_enu(
                    gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt
                )

                merged["target_e"] = e
                merged["target_n"] = n

                # Filter Invalid Targets (NaN, Inf, or Massive Outliers)
                # Cite debug_lesson_11: Filter for Finiteness (Inf), Not Just Missing Values (NaN)
                valid_mask = np.isfinite(merged["target_e"]) & np.isfinite(
                    merged["target_n"]
                )

                # Cite debug_lesson_25: Avoid Indiscriminate Zero-Filling on Coordinate Reference Data
                # If WLS is 0 (center of earth) or invalid, error will be huge (> 6000km).
                # We filter out physical outliers (> 1000km) to prevent gradient explosion.
                range_mask = (merged["target_e"].abs() < 1e6) & (
                    merged["target_n"].abs() < 1e6
                )

                final_mask = valid_mask & range_mask

                dropped_count = len(merged) - final_mask.sum()
                if dropped_count > 0:
                    print(
                        f"Drive {drive_id}: Dropped {dropped_count} rows due to invalid targets."
                    )
                    merged = merged[final_mask].copy()

                if merged.empty:
                    return None

                return merged
            else:
                print(f"GT file not found: {full_gt_path}")
                return None
        else:
            # Test set (no targets)
            return features

    def process_data(self, metadata_path, load_cached_data=True, split="train"):
        """
        Main entry point to process data for a specific split.
        """
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(self.config.CACHE_DIR, f"{split}_processed.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {split} data from {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Processing {split} data from scratch...")

        df_meta = pd.read_csv(metadata_path)

        # Get unique trips to process
        unique_trips = df_meta[
            ["drive_id", "phone_name", "gnss_path"]
        ].drop_duplicates()

        processed_dfs = []

        for _, row in unique_trips.iterrows():
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]
            gnss_path = row["gnss_path"]

            # Determine GT path based on split
            if split in ["train", "val"]:
                # In the provided dataset structure, ground_truth.csv is in the same folder as device_gnss.csv
                gt_path = gnss_path.replace("device_gnss.csv", "ground_truth.csv")
            else:
                gt_path = None

            df = self._process_drive(drive_id, phone_name, gnss_path, gt_path)
            if df is not None and not df.empty:
                processed_dfs.append(df)

        if not processed_dfs:
            raise ValueError(f"No data processed for split {split}")

        final_df = pd.concat(processed_dfs, ignore_index=True)

        # Sort by drive and time to ensure sequential order for time-series modeling
        final_df = final_df.sort_values(
            ["drive_id", "phone_name", "UnixTimeMillis"]
        ).reset_index(drop=True)

        # Save cache
        print(f"Saving {split} data to {cache_file}")
        final_df.to_parquet(cache_file, index=False)

        return final_df
