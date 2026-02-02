import os
import numpy as np
import pandas as pd
import warnings
from library.config import Config
from library.utils import GeodeticUtils

# Suppress warnings
warnings.filterwarnings("ignore")


class GNSSPreprocessor:
    """
    Handles loading, cleaning, feature engineering, and target generation for GNSS data.
    """

    def __init__(self, config: Config):
        self.config = config

    def quantize_time(self, df, time_col="UnixTimeMillis"):
        """
        Rounds timestamps to the nearest second (1000 ms) to align disparate logs.
        """
        df[time_col] = np.round(df[time_col] / 1000) * 1000
        df[time_col] = df[time_col].astype(np.int64)
        return df

    def get_phase_validity(self, adr_state):
        """
        Checks if the Carrier Phase is valid based on the AccumulatedDeltaRangeState.
        Bit 0 represents Valid.
        """
        # Bitwise AND with 1 (2^0)
        return (adr_state.astype(int) & self.config.ADR_STATE_VALID_BIT) > 0

    def stratify_and_aggregate(self, gnss_df):
        """
        Performs stratified aggregation of GNSS signals for each epoch.
        Returns a DataFrame with one row per timestamp and flattened feature columns.
        """
        # Ensure timestamp is quantized
        if "UnixTimeMillis" not in gnss_df.columns:
            # device_gnss.csv usually has utcTimeMillis
            gnss_df["UnixTimeMillis"] = gnss_df["utcTimeMillis"]

        gnss_df = self.quantize_time(gnss_df, "UnixTimeMillis")

        # Pre-calculate derived columns
        gnss_df["is_L5"] = gnss_df["SignalType"].isin(self.config.L5_SIGNAL_TYPES)
        gnss_df["is_LowElev"] = (
            gnss_df["SvElevationDegrees"] < self.config.HIGH_RISK_ELEV_THRESHOLD
        )
        gnss_df["PhaseValid"] = self.get_phase_validity(
            gnss_df["AccumulatedDeltaRangeState"].fillna(0)
        )

        # Define Strata Masks
        # We will process groupings manually to ensure all strata are present
        # 1. Global (All signals)
        # 2. HighPrecision (L5)
        # 3. HighRisk (Low Elev)

        # We'll collect aggregated features in a list of DataFrames to merge later
        aggregated_dfs = []

        # Group by Timestamp
        grouped = gnss_df.groupby("UnixTimeMillis")

        # --- 1. Global Context Features ---
        # Signal Count
        global_ctx = grouped.size().to_frame(name="Global_SignalCount")

        # RawPseudorangeUncertaintyMeters Mean
        global_ctx["Global_RawPseudorangeUncertaintyMeters_mean"] = grouped[
            "RawPseudorangeUncertaintyMeters"
        ].mean()

        # Azimuth Centroid (Geometric Center of Constellation)
        # Convert degrees to radians
        az_rad = np.deg2rad(gnss_df["SvAzimuthDegrees"].fillna(0))
        gnss_df["az_cos"] = np.cos(az_rad)
        gnss_df["az_sin"] = np.sin(az_rad)

        # Re-group for azimuth
        grouped_az = gnss_df.groupby("UnixTimeMillis")
        global_ctx["Global_AzimuthCentroid_X"] = grouped_az["az_cos"].mean()
        global_ctx["Global_AzimuthCentroid_Y"] = grouped_az["az_sin"].mean()

        aggregated_dfs.append(global_ctx)

        # --- 2. Stratified Features ---
        strata_definitions = {
            "Global": pd.Series(True, index=gnss_df.index),
            "HighPrecision": gnss_df["is_L5"],
            "HighRisk": gnss_df["is_LowElev"],
        }

        for stratum_name, mask in strata_definitions.items():
            # Filter data for this stratum
            stratum_data = gnss_df[mask]

            if stratum_data.empty:
                # If stratum is empty for the entire drive, create placeholder with index
                # We need the index from global_ctx to ensure alignment
                empty_df = pd.DataFrame(index=global_ctx.index)
                for feat in ["Cn0DbHz", "SvElevationDegrees"]:
                    for stat in self.config.STATS_FUNCS:
                        empty_df[f"{stratum_name}_{feat}_{stat}"] = 0.0
                empty_df[f"{stratum_name}_PhaseValid_count"] = 0.0
                empty_df[f"{stratum_name}_PhaseValid_fraction"] = 0.0
                aggregated_dfs.append(empty_df)
                continue

            stratum_grouped = stratum_data.groupby("UnixTimeMillis")

            # Aggregations for Cn0 and Elevation
            stats_df = stratum_grouped[["Cn0DbHz", "SvElevationDegrees"]].agg(
                self.config.STATS_FUNCS
            )

            # Flatten MultiIndex columns
            stats_df.columns = [
                f"{stratum_name}_{c[0]}_{c[1]}" for c in stats_df.columns
            ]

            # Phase Validity Features
            phase_df = stratum_grouped["PhaseValid"].agg(["sum", "mean"])
            phase_df.columns = [
                f"{stratum_name}_PhaseValid_count",
                f"{stratum_name}_PhaseValid_fraction",
            ]

            # Merge stats and phase info
            stratum_features = pd.concat([stats_df, phase_df], axis=1)
            aggregated_dfs.append(stratum_features)

        # --- 3. Merge All Features ---
        # Concatenate along columns, aligning on UnixTimeMillis index
        final_df = pd.concat(aggregated_dfs, axis=1)

        # Fill NaNs (e.g. if a timestamp has global signals but no L5 signals, L5 cols will be NaN)
        final_df = final_df.fillna(0.0)

        return final_df.reset_index()

    def create_targets(self, features_df, gt_df, wls_df):
        """
        Computes regression targets: deviation of Ground Truth from WLS Baseline in ENU meters.
        """
        # 1. Prepare WLS Baseline
        # WLS positions are in ECEF in device_gnss.csv. We need them per timestamp.
        # Since WLS is computed per epoch, it's repeated for every signal. We take the first.
        if "UnixTimeMillis" not in wls_df.columns:
            wls_df["UnixTimeMillis"] = wls_df["utcTimeMillis"]
        wls_df = self.quantize_time(wls_df, "UnixTimeMillis")

        # Extract unique WLS positions per timestamp
        wls_baseline = wls_df.groupby("UnixTimeMillis").first().reset_index()

        # Convert WLS ECEF to Lat/Lon
        # Columns: WlsPositionXEcefMeters, WlsPositionYEcefMeters, WlsPositionZEcefMeters
        wls_lat, wls_lon, wls_alt = GeodeticUtils.ecef_to_geodetic(
            wls_baseline["WlsPositionXEcefMeters"].values,
            wls_baseline["WlsPositionYEcefMeters"].values,
            wls_baseline["WlsPositionZEcefMeters"].values,
        )

        wls_baseline["WLS_Lat"] = wls_lat
        wls_baseline["WLS_Lon"] = wls_lon
        wls_baseline["WLS_Alt"] = wls_alt

        # 2. Prepare Ground Truth
        gt_df = self.quantize_time(gt_df, "UnixTimeMillis")
        # GT columns: LatitudeDegrees, LongitudeDegrees, AltitudeMeters

        # 3. Merge Features, WLS, and GT
        # Inner join to ensure we only train on timestamps where we have everything
        merged = pd.merge(
            features_df,
            wls_baseline[["UnixTimeMillis", "WLS_Lat", "WLS_Lon", "WLS_Alt"]],
            on="UnixTimeMillis",
            how="inner",
        )
        merged = pd.merge(
            merged,
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

        # 4. Compute Targets (ENU Residuals)
        # We calculate the vector FROM WLS TO GT
        # Reference point is WLS position
        e, n, u = GeodeticUtils.wgs84_to_enu(
            merged["LatitudeDegrees"].values,
            merged["LongitudeDegrees"].values,
            merged["AltitudeMeters"].values,
            merged["WLS_Lat"].values,
            merged["WLS_Lon"].values,
            merged["WLS_Alt"].values,
        )

        merged["target_E"] = e
        merged["target_N"] = n

        # Keep WLS coordinates for reconstruction during inference/validation
        # Drop GT columns to prevent leakage (though we return X, y usually)

        return merged

    def process_drive(self, drive_id, phone_name, gnss_rel_path, gt_df=None):
        """
        Processes a single drive: loads data, extracts features, and computes targets if GT is provided.
        """
        gnss_path = os.path.join(self.config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found: {gnss_path}")
            return None

        # Load GNSS
        try:
            gnss_df = pd.read_csv(gnss_path)
        except Exception as e:
            print(f"Error reading {gnss_path}: {e}")
            return None

        # Feature Engineering
        features_df = self.stratify_and_aggregate(gnss_df)

        # Add Metadata
        features_df["drive_id"] = drive_id
        features_df["phone_name"] = phone_name

        # If Ground Truth is provided (Train/Val mode), compute targets
        if gt_df is not None:
            # We need WLS info from gnss_df to compute targets
            # Reuse gnss_df as wls_source
            final_df = self.create_targets(features_df, gt_df, gnss_df)
        else:
            # Test mode: We still need WLS baseline for reconstruction later
            if "UnixTimeMillis" not in gnss_df.columns:
                gnss_df["UnixTimeMillis"] = gnss_df["utcTimeMillis"]
            gnss_df = self.quantize_time(gnss_df, "UnixTimeMillis")

            wls_baseline = gnss_df.groupby("UnixTimeMillis").first().reset_index()
            wls_lat, wls_lon, wls_alt = GeodeticUtils.ecef_to_geodetic(
                wls_baseline["WlsPositionXEcefMeters"].values,
                wls_baseline["WlsPositionYEcefMeters"].values,
                wls_baseline["WlsPositionZEcefMeters"].values,
            )
            wls_baseline["WLS_Lat"] = wls_lat
            wls_baseline["WLS_Lon"] = wls_lon
            wls_baseline["WLS_Alt"] = wls_alt

            final_df = pd.merge(
                features_df,
                wls_baseline[["UnixTimeMillis", "WLS_Lat", "WLS_Lon", "WLS_Alt"]],
                on="UnixTimeMillis",
                how="inner",
            )

        return final_df

    def process_dataset(self, metadata_path, mode="train", load_cached_data=True):
        """
        Main entry point to process a full dataset defined by a metadata CSV.
        Handles caching.
        """
        cache_file = os.path.join(self.config.CACHE_DIR, f"{mode}_processed.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {mode} data from {cache_file}...")
            return pd.read_parquet(cache_file)

        print(f"Processing {mode} data from scratch...")
        meta_df = pd.read_csv(metadata_path)

        # Optional: Subsample for debugging
        if self.config.DEBUG:
            drives = meta_df["drive_id"].unique()[: self.config.DEBUG_DRIVE_COUNT]
            meta_df = meta_df[meta_df["drive_id"].isin(drives)]
            print(f"DEBUG MODE: Processing only {len(drives)} drives.")

        processed_dfs = []

        # Iterate over unique phone runs (drive + phone)
        # In train_metadata, we have GT info. In test, we don't.
        # We group by drive_id and phone_name to process each sequence.

        unique_runs = meta_df.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group in unique_runs:
            # Get paths from the first row of the group
            row = group.iloc[0]
            gnss_path = row["gnss_path"]

            gt_df = None
            if mode in ["train", "val"]:
                # Construct GT DataFrame from the metadata group
                # The metadata file IS the ground truth for train/val
                gt_df = group[
                    [
                        "UnixTimeMillis",
                        "LatitudeDegrees",
                        "LongitudeDegrees",
                        "AltitudeMeters",
                    ]
                ].copy()

            df = self.process_drive(drive_id, phone_name, gnss_path, gt_df)

            if df is not None and not df.empty:
                processed_dfs.append(df)

        if not processed_dfs:
            raise ValueError(f"No data processed for mode {mode}. Check input paths.")

        full_df = pd.concat(processed_dfs, ignore_index=True)

        # Save to cache
        print(f"Saving {mode} data to {cache_file}...")
        full_df.to_parquet(cache_file, index=False)

        return full_df
