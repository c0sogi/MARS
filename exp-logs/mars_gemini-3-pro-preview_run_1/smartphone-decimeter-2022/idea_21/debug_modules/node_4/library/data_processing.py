import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, ecef_to_wgs84, wgs84_to_enu


def process_gnss_log(gnss_path):
    """
    Process a single GNSS log file to extract aggregated features.
    Aligns timestamps to 1Hz and computes stats for L1 and L5 signals.
    """
    try:
        df = pd.read_csv(gnss_path)
    except FileNotFoundError:
        return None

    # Filter out rows with invalid timestamps or signals if necessary
    # For now, we assume the provided csv is reasonably clean but we align timestamps

    # Define signal groups based on Config
    is_l1 = df["SignalType"].isin(Config.SIGNAL_TYPES_L1)
    is_l5 = df["SignalType"].isin(Config.SIGNAL_TYPES_L5)

    # Pre-calculate Azimuth components (Unit Vectors)
    # Azimuth is degrees clockwise from North.
    rad_az = np.deg2rad(df["SvAzimuthDegrees"].fillna(0))
    df["Az_X"] = np.sin(rad_az)
    df["Az_Y"] = np.cos(rad_az)

    # Helper to aggregate statistics for a specific signal mask
    def agg_group(mask, prefix):
        group_df = df[mask]
        if group_df.empty:
            # Return empty dataframe with correct columns to ensure schema consistency
            cols = []
            for feat in Config.STAT_FEATURES:
                cols.extend(
                    [
                        f"{prefix}_{feat}_{stat}"
                        for stat in ["mean", "std", "min", "max"]
                    ]
                )
            return pd.DataFrame(columns=cols)

        agg_dict = {}
        for feat in Config.STAT_FEATURES:
            agg_dict[feat] = ["mean", "std", "min", "max"]

        # Group by timestamp (utcTimeMillis)
        grouped = group_df.groupby("utcTimeMillis").agg(agg_dict)

        # Flatten MultiIndex columns
        grouped.columns = [f"{prefix}_{c[0]}_{c[1]}" for c in grouped.columns]
        return grouped

    # Compute aggregations for L1 and L5 bands
    l1_agg = agg_group(is_l1, "L1")
    l5_agg = agg_group(is_l5, "L5")

    # Global aggregations (SatCount, Azimuth Centroid, WLS Baseline)
    # Use mean for WLS to avoid NaN from first row (Cite debug_lesson_6)
    global_agg_dict = {
        "Svid": "count",  # Satellite Count
        "RawPseudorangeUncertaintyMeters": "mean",
        "Az_X": "mean",
        "Az_Y": "mean",
        "WlsPositionXEcefMeters": "mean",
        "WlsPositionYEcefMeters": "mean",
        "WlsPositionZEcefMeters": "mean",
    }

    # Ensure columns exist before aggregating
    for col in list(global_agg_dict.keys()):
        if col not in df.columns and col != "Svid":  # Svid is always there
            # If WLS columns missing, fill with NaN to detect later
            df[col] = np.nan

    global_agg = df.groupby("utcTimeMillis").agg(global_agg_dict)
    global_agg.rename(columns={"Svid": "SatCount"}, inplace=True)

    # Merge all features
    # Join on index (utcTimeMillis)
    result = global_agg.join(l1_agg, how="left").join(l5_agg, how="left")

    # Selective Fillna (Cite debug_lesson_25)
    # Identify WLS columns
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Fill features with 0
    feature_cols = [c for c in result.columns if c not in wls_cols]
    result[feature_cols] = result[feature_cols].fillna(0)

    # Drop rows where WLS is missing (cannot compute target)
    result = result.dropna(subset=wls_cols)

    # Cite debug_lesson_11: Filter for Finiteness (Inf), Not Just Missing Values (NaN)
    # Only replace Inf in feature columns, WLS should be finite or dropped
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], 0)

    # Drop rows with Inf in WLS columns
    mask_wls_finite = np.isfinite(result[wls_cols]).all(axis=1)
    result = result[mask_wls_finite]

    return result


def get_ground_truth_with_altitude(drive_id, phone_name, input_dir):
    """
    Reads the ground truth file directly to get Altitude, which is not present in the
    metadata CSVs but is required for accurate ECEF->ENU conversion.
    """
    gt_path = os.path.join(input_dir, "train", drive_id, phone_name, "ground_truth.csv")
    if os.path.exists(gt_path):
        return pd.read_csv(gt_path)
    return None


def process_dataset(metadata_path, mode="train", load_cached_data=True):
    """
    Main data processing function.
    Loads metadata, processes raw GNSS files, computes targets (for train/val),
    and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataset ready for the model.
    """
    set_seed()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_processed.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {mode} data from scratch...")

    df_meta = pd.read_csv(metadata_path)

    # Identify unique trips (drive + phone) to process sequentially
    # This avoids repeated file I/O if metadata is shuffled
    if "tripId" in df_meta.columns:
        # Test metadata has tripId
        unique_trips = df_meta[["drive_id", "phone_name"]].drop_duplicates().values
    else:
        # Train metadata
        unique_trips = df_meta[["drive_id", "phone_name"]].drop_duplicates().values

    processed_frames = []

    for drive_id, phone_name in unique_trips:
        # Locate the GNSS file path from metadata
        # We take the first row for this trip to get the path
        trip_subset = df_meta[
            (df_meta["drive_id"] == drive_id) & (df_meta["phone_name"] == phone_name)
        ]
        if trip_subset.empty:
            continue

        row = trip_subset.iloc[0]
        gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

        # Extract Features
        gnss_features = process_gnss_log(gnss_path)

        if gnss_features is None:
            print(f"Warning: GNSS file not found for {drive_id} {phone_name}")
            continue

        # Reset index to make utcTimeMillis a column, rename to match GT
        gnss_features = gnss_features.reset_index()
        gnss_features.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

        # Add ID columns
        gnss_features["drive_id"] = drive_id
        gnss_features["phone_name"] = phone_name

        # Timestamp Rounding for Merge (Cite debug_lesson_4)
        gnss_features["UnixTimeMillis_rounded"] = (
            np.round(gnss_features["UnixTimeMillis"] / 1000) * 1000
        )

        # Merge logic depends on mode
        if mode in ["train", "val"]:
            # Load Ground Truth with Altitude
            gt_df = get_ground_truth_with_altitude(
                drive_id, phone_name, Config.INPUT_DIR
            )
            if gt_df is None:
                continue

            # Round GT timestamps
            gt_df["UnixTimeMillis_rounded"] = (
                np.round(gt_df["UnixTimeMillis"] / 1000) * 1000
            )

            # Merge Features with GT on Timestamp
            # Inner join ensures we only keep rows where we have both features and labels
            merged = pd.merge(
                gt_df,
                gnss_features,
                on="UnixTimeMillis_rounded",
                how="inner",
                suffixes=("", "_gnss"),
            )

            # Compute Targets: ENU Residuals
            # 1. Convert WLS ECEF to WLS Lat/Lon/Alt (Reference Position)
            wls_x = merged["WlsPositionXEcefMeters"].values
            wls_y = merged["WlsPositionYEcefMeters"].values
            wls_z = merged["WlsPositionZEcefMeters"].values

            ref_lat, ref_lon, ref_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

            # 2. Get GT Lat/Lon/Alt
            gt_lat = merged["LatitudeDegrees"].values
            gt_lon = merged["LongitudeDegrees"].values
            gt_alt = merged["AltitudeMeters"].values

            # 3. Calculate ENU offsets (Target = GT - Reference)
            e, n, u = wgs84_to_enu(gt_lat, gt_lon, gt_alt, ref_lat, ref_lon, ref_alt)

            merged["Target_E"] = e
            merged["Target_N"] = n
            merged["Target_U"] = u

            # Store WLS Lat/Lon for debugging or reconstruction verification
            merged["Wls_Lat"] = ref_lat
            merged["Wls_Lon"] = ref_lon
            merged["Wls_Alt"] = ref_alt

            # Cite debug_lesson_11: Filter for Finiteness (Inf), Not Just Missing Values (NaN)
            # Sanitize Targets: Drop rows with NaN/Inf in Targets
            # Also drop rows with unrealistically large targets (e.g., missing WLS baseline -> 0,0,0)
            mask_finite = np.isfinite(merged[["Target_E", "Target_N"]]).all(axis=1)
            mask_range = (merged["Target_E"].abs() < 1e6) & (
                merged["Target_N"].abs() < 1e6
            )

            merged = merged[mask_finite & mask_range]

            # Cite debug_lesson_5: Validate Data Volume After Filtering
            if merged.empty:
                continue

            # Select relevant columns (Features + Targets + IDs)
            # We keep all columns for now, will filter in Dataset class
            processed_frames.append(merged)

        else:
            # Test Mode
            # We need to output predictions for specific timestamps in metadata
            # Left join metadata with features to ensure we have rows for all required timestamps

            # Filter metadata for this trip only to avoid duplicates
            trip_meta_subset = df_meta[
                (df_meta["drive_id"] == drive_id)
                & (df_meta["phone_name"] == phone_name)
            ].copy()

            trip_meta_subset["UnixTimeMillis_rounded"] = (
                np.round(trip_meta_subset["UnixTimeMillis"] / 1000) * 1000
            )

            merged = pd.merge(
                trip_meta_subset,
                gnss_features,
                on=["drive_id", "phone_name", "UnixTimeMillis_rounded"],
                how="left",
                suffixes=("", "_gnss"),
            )

            # Handle missing GNSS data (timestamps in submission not in GNSS log)
            # Forward fill then backward fill to propagate nearby features
            # Only fill feature columns, not IDs
            feature_cols = [
                c
                for c in gnss_features.columns
                if c
                not in [
                    "drive_id",
                    "phone_name",
                    "UnixTimeMillis",
                    "UnixTimeMillis_rounded",
                ]
            ]
            merged[feature_cols] = merged[feature_cols].ffill().bfill().fillna(0)

            # Calculate WLS Reference Lat/Lon/Alt for reconstruction
            wls_x = merged["WlsPositionXEcefMeters"].values
            wls_y = merged["WlsPositionYEcefMeters"].values
            wls_z = merged["WlsPositionZEcefMeters"].values

            # Handle case where WLS might be 0 (if filled by fillna(0))
            # This shouldn't happen often if logs are aligned, but for safety:
            # If WLS is 0, ref_lat/lon will be 0.
            ref_lat, ref_lon, ref_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

            merged["Wls_Lat"] = ref_lat
            merged["Wls_Lon"] = ref_lon
            merged["Wls_Alt"] = ref_alt

            processed_frames.append(merged)

    if not processed_frames:
        print("Error: No data processed.")
        return pd.DataFrame()

    final_df = pd.concat(processed_frames, ignore_index=True)

    # Save to cache
    final_df.to_parquet(cache_file)
    print(f"Saved processed data to {cache_file} (Shape: {final_df.shape})")

    return final_df
