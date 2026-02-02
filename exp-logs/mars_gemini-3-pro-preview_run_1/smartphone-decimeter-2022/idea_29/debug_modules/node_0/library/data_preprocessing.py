import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.utils import (
    wgs84_to_ecef,
    ecef_to_wgs84,
    ecef_to_enu,
    calculate_azimuth_centroid,
)


def load_gnss_log(path):
    """
    Loads the device_gnss.csv file.
    """
    # Columns required for feature engineering and baseline extraction
    use_cols = [
        "utcTimeMillis",
        "SignalType",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "Cn0DbHz",
        "AccumulatedDeltaRangeState",
        "RawPseudorangeUncertaintyMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"GNSS file not found: {path}")

    df = pd.read_csv(path, usecols=lambda c: c in use_cols)

    # Rename utcTimeMillis to UnixTimeMillis for consistency
    df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    return df


def align_timestamps(df):
    """
    Rounds timestamps to the nearest second (1Hz alignment).
    """
    df["UnixTimeMillis"] = (np.round(df["UnixTimeMillis"] / 1000) * 1000).astype(
        np.int64
    )
    return df


def compute_features(gnss_df):
    """
    Performs Phase-Aware Stratified Aggregation and Global Context feature extraction.
    Returns a DataFrame with one row per unique timestamp.
    """
    # 1. Pre-calculate weights and components for Azimuth Centroid
    # We use Cn0 as weight for azimuth centroid
    gnss_df["Az_Sin_W"] = (
        np.sin(np.radians(gnss_df["SvAzimuthDegrees"])) * gnss_df["Cn0DbHz"]
    )
    gnss_df["Az_Cos_W"] = (
        np.cos(np.radians(gnss_df["SvAzimuthDegrees"])) * gnss_df["Cn0DbHz"]
    )

    # 2. Define Strata Masks
    # Stratum 2: High-Integrity
    # SignalType in High-Integrity Set OR Valid Carrier Phase (Bit 0 of ADR State)
    is_high_integrity_signal = gnss_df["SignalType"].isin(Config.HIGH_INTEGRITY_SIGNALS)
    is_phase_valid = (
        gnss_df["AccumulatedDeltaRangeState"] & Config.ADR_STATE_VALID_BIT
    ) != 0
    mask_high_int = is_high_integrity_signal | is_phase_valid

    # Stratum 3: Low-Quality
    # Elevation < Threshold OR Cn0 < Threshold
    mask_low_qual = (gnss_df["SvElevationDegrees"] < Config.LOW_QUALITY_ELEV_TH) | (
        gnss_df["Cn0DbHz"] < Config.LOW_QUALITY_CN0_TH
    )

    # Stratum 1: Global (All) - Implicit

    # 3. Aggregation
    # We process each stratum separately and then merge

    # Helper to aggregate a subset
    def agg_stratum(subset, prefix):
        if subset.empty:
            # Return DataFrame with 0s/NaNs if stratum is empty for the whole drive (unlikely but possible)
            # We need to ensure columns exist. We'll handle this by reindexing later if needed.
            return pd.DataFrame()

        agg_funcs = {
            "Cn0DbHz": Config.STRATUM_STATS_OPS,
            "SvElevationDegrees": Config.STRATUM_STATS_OPS,
        }

        # Group by Time
        grouped = subset.groupby("UnixTimeMillis")
        res = grouped.agg(agg_funcs)

        # Flatten MultiIndex columns: e.g., Cn0DbHz_mean -> HighInt_Cn0DbHz_mean
        res.columns = [f"{prefix}_{c[0]}_{c[1]}" for c in res.columns]
        return res

    # Aggregate Strata
    feat_global = agg_stratum(gnss_df, "Global")
    feat_high = agg_stratum(gnss_df[mask_high_int], "HighInt")
    feat_low = agg_stratum(gnss_df[mask_low_qual], "LowQual")

    # 4. Global Context Features
    # SatCount, Mean Uncertainty, Azimuth Centroid
    grouped_all = gnss_df.groupby("UnixTimeMillis")
    context = grouped_all.agg(
        {
            "SignalType": "count",  # SatCount
            "RawPseudorangeUncertaintyMeters": "mean",
            "Az_Sin_W": "sum",
            "Az_Cos_W": "sum",
            "Cn0DbHz": "sum",  # Sum of weights for normalization
        }
    )

    context.rename(columns={"SignalType": "SatCount"}, inplace=True)

    # Calculate Centroids
    # Avoid division by zero
    w_sum = context["Cn0DbHz"].replace(0, 1.0)
    context["Azimuth_Sin"] = context["Az_Sin_W"] / w_sum
    context["Azimuth_Cos"] = context["Az_Cos_W"] / w_sum

    # Drop temp cols
    context.drop(columns=["Az_Sin_W", "Az_Cos_W", "Cn0DbHz"], inplace=True)

    # 5. Merge All Features
    # Base is context (contains all timestamps)
    features = context.join([feat_global, feat_high, feat_low], how="left")

    # Fill NaNs for missing strata (e.g. no LowQual sats at a timestamp)
    # For counts/stats, 0 is often a safe impute for neural nets if standardized,
    # but for Min/Max it might be misleading.
    # However, standard practice in this domain for missing signals is 0 padding.
    features.fillna(0, inplace=True)

    return features


def compute_targets(features_df, gt_df, gnss_df):
    """
    Computes regression targets: Delta East, Delta North (Meters)
    relative to the WLS baseline.
    """
    # 1. Get WLS Baseline per timestamp
    # We take the first WLS position per timestamp (they are repeated per sat)
    wls_ref = gnss_df.groupby("UnixTimeMillis").first()[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ]

    # 2. Merge Features with Ground Truth and WLS
    # Inner join ensures we only compute targets where we have GT
    merged = features_df.join(wls_ref, how="inner")
    merged = merged.merge(
        gt_df[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on="UnixTimeMillis",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    # 3. Convert WLS ECEF to WLS LLA (to get a reference altitude and for ENU conversion)
    wls_x = merged["WlsPositionXEcefMeters"].values
    wls_y = merged["WlsPositionYEcefMeters"].values
    wls_z = merged["WlsPositionZEcefMeters"].values

    wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

    # 4. Convert GT LLA to ECEF
    # We use WLS Altitude as proxy for GT Altitude since GT Altitude is missing in metadata
    # This approximation is sufficient for horizontal error calculation.
    gt_lat = merged["LatitudeDegrees"].values
    gt_lon = merged["LongitudeDegrees"].values

    gt_x, gt_y, gt_z = wgs84_to_ecef(gt_lat, gt_lon, wls_alt)

    # 5. Compute ENU Residuals (Target = GT - WLS)
    # We calculate the position of GT relative to WLS in the ENU frame centered at WLS
    # This vector points FROM WLS TO GT.
    # So if Model predicts (dE, dN), then Pred_Pos = WLS + (dE, dN)
    d_east, d_north, _ = ecef_to_enu(gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt)

    # Assign targets
    merged["Target_E"] = d_east
    merged["Target_N"] = d_north

    # Keep only features and targets
    cols_to_keep = (
        ["UnixTimeMillis"] + Config.get_feature_names() + ["Target_E", "Target_N"]
    )

    # Filter columns that exist (in case some strata were totally missing)
    existing_cols = [c for c in cols_to_keep if c in merged.columns]

    return merged[existing_cols]


def process_drive(drive_id, phone_name, gnss_path, gt_df=None):
    """
    Processes a single drive: loads data, aligns, extracts features, and computes targets.
    """
    full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)

    try:
        # 1. Load Raw Data
        gnss_df = load_gnss_log(full_gnss_path)

        # 2. Align Timestamps
        gnss_df = align_timestamps(gnss_df)

        # 3. Feature Engineering
        features = compute_features(gnss_df)

        # 4. Target Generation (if GT exists)
        if gt_df is not None and not gt_df.empty:
            # Align GT timestamps
            gt_df = align_timestamps(gt_df.copy())
            processed_df = compute_targets(features, gt_df, gnss_df)
        else:
            # Test mode: Just features
            # We still need WLS positions for the submission/inference stage reconstruction,
            # but for the model input (X), we only need features.
            # However, to keep track of WLS for later, we might want to include them or just
            # rely on the fact that we can reload them.
            # For this module, we return features.
            processed_df = features.reset_index()  # UnixTimeMillis becomes a column

            # Add dummy targets for consistency if needed, or just handle in dataset class
            processed_df["Target_E"] = np.nan
            processed_df["Target_N"] = np.nan

            # Filter cols
            cols = (
                ["UnixTimeMillis"]
                + Config.get_feature_names()
                + ["Target_E", "Target_N"]
            )
            # Ensure all feature columns exist (fill 0 if missing due to empty strata)
            for c in Config.get_feature_names():
                if c not in processed_df.columns:
                    processed_df[c] = 0.0

            processed_df = processed_df[cols]

        # Add Metadata
        processed_df["drive_id"] = drive_id
        processed_df["phone_name"] = phone_name

        return processed_df

    except Exception as e:
        print(f"Error processing drive {drive_id}/{phone_name}: {e}")
        return pd.DataFrame()


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Main entry point for processing a dataset (Train, Val, or Test).
    Handles caching.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # 3. Process Drives
    results = []

    # Group by drive and phone to process each sequence
    # For Test data, metadata has one row per timestamp, so we need to get unique drive-phone pairs
    unique_drives = meta_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    print(f"Processing {len(unique_drives)} drives from {metadata_path}...")

    for _, row in tqdm(unique_drives.iterrows(), total=len(unique_drives)):
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        # Get GT for this drive if available (Train/Val)
        # Check if metadata has target columns
        if "LatitudeDegrees" in meta_df.columns:
            drive_gt = meta_df[
                (meta_df["drive_id"] == drive_id)
                & (meta_df["phone_name"] == phone_name)
            ].copy()
        else:
            drive_gt = None

        drive_data = process_drive(drive_id, phone_name, gnss_path, drive_gt)

        if not drive_data.empty:
            results.append(drive_data)

    if not results:
        print("Warning: No data processed.")
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)

    # 4. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    final_df.to_parquet(cache_path, index=False)

    return final_df
