import os
import pandas as pd
import numpy as np
from library.config import INPUT_DIR, OUTPUT_DIR, process_gnss_data, aggregate_features
from library.utils import ecef_to_lla, lla_to_enu


def load_drive_data(
    drive_id, phone_name, gnss_rel_path, imu_rel_path, gt_rel_path=None
):
    """
    Loads raw CSV data for a specific drive and phone.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        gnss_rel_path (str): Relative path to GNSS file.
        imu_rel_path (str): Relative path to IMU file.
        gt_rel_path (str, optional): Relative path to Ground Truth file.

    Returns:
        tuple: (gnss_df, imu_df, gt_df) DataFrames. gt_df is None if path not provided.
    """
    gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)
    imu_path = os.path.join(INPUT_DIR, imu_rel_path)

    if not os.path.exists(gnss_path):
        print(f"Warning: GNSS file missing at {gnss_path}")
        return None, None, None

    gnss_df = pd.read_csv(gnss_path)

    imu_df = None
    if os.path.exists(imu_path):
        imu_df = pd.read_csv(imu_path)

    gt_df = None
    if gt_rel_path:
        gt_path = os.path.join(INPUT_DIR, gt_rel_path)
        if os.path.exists(gt_path):
            gt_df = pd.read_csv(gt_path)

    return gnss_df, imu_df, gt_df


def align_timestamps(features_df, targets_df, on_col="UnixTimeMillis", how="inner"):
    """
    Aligns high-frequency/irregular features with target timestamps.

    Args:
        features_df (pd.DataFrame): Processed features indexed by timestamp.
        targets_df (pd.DataFrame): Metadata/Targets containing the required timestamps.
        on_col (str): Column name for timestamp in targets_df.
        how (str): Type of merge to be performed.

    Returns:
        pd.DataFrame: Merged dataframe containing features aligned to targets.
    """
    # Ensure features_df index is named correctly for joining
    if features_df.index.name != on_col:
        features_df.index.name = on_col

    # Set index on targets for efficient join
    targets_indexed = targets_df.set_index(on_col)

    # Join based on the specified method
    merged = targets_indexed.join(features_df, how=how)

    return merged


def compute_targets(df):
    """
    Computes ENU residuals (targets) and WLS LLA coordinates.

    Args:
        df (pd.DataFrame): Dataframe containing WLS ECEF positions and GT Lat/Lon.

    Returns:
        pd.DataFrame: Dataframe with added 'target_E', 'target_N', 'wls_lat', 'wls_lon'.
    """
    # Extract WLS ECEF coordinates
    x = df["WlsPositionXEcefMeters"].values
    y = df["WlsPositionYEcefMeters"].values
    z = df["WlsPositionZEcefMeters"].values

    # Convert WLS ECEF to LLA
    wls_lat, wls_lon, wls_alt = ecef_to_lla(x, y, z)

    df["wls_lat"] = wls_lat
    df["wls_lon"] = wls_lon
    df["wls_alt"] = wls_alt

    # Check if we have Ground Truth coordinates
    if "LatitudeDegrees" in df.columns and "LongitudeDegrees" in df.columns:
        gt_lat = df["LatitudeDegrees"].values
        gt_lon = df["LongitudeDegrees"].values

        # Use GT Altitude if available, else fallback to WLS Altitude
        # This minimizes error in ENU projection due to altitude mismatch
        if "AltitudeMeters" in df.columns:
            gt_alt = df["AltitudeMeters"].fillna(df["wls_alt"]).values
        else:
            gt_alt = wls_alt

        # Compute ENU residuals: Vector from WLS (Ref) to GT (Target)
        e, n, u = lla_to_enu(gt_lat, gt_lon, gt_alt, wls_lat, wls_lon, wls_alt)

        df["target_E"] = e
        df["target_N"] = n
        # We don't predict Up, but it's part of the transformation

    return df


def get_dataset(metadata_path, load_cached_data=True, split="train"):
    """
    Main function to load, process, and cache the dataset.

    Args:
        metadata_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to load from cache if available.
        split (str): Dataset split name ('train', 'val', 'test').

    Returns:
        pd.DataFrame: The processed dataset ready for training/inference.
    """
    cache_file = os.path.join(OUTPUT_DIR, f"{split}_features.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Processing {split} data from scratch...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    features_list = []
    unique_drives = meta_df["drive_id"].unique()

    # Determine join method based on split (Cite debug_lesson_4)
    join_method = "left" if split == "test" else "inner"

    for drive_id in unique_drives:
        drive_meta = meta_df[meta_df["drive_id"] == drive_id]

        for phone_name in drive_meta["phone_name"].unique():
            subset = drive_meta[drive_meta["phone_name"] == phone_name]
            if subset.empty:
                continue

            # Extract paths from the first row of the subset
            first_row = subset.iloc[0]
            gnss_rel = first_row["gnss_path"]
            imu_rel = first_row["imu_path"]
            gt_rel = first_row.get("gt_path")  # May be None for test

            # Load Raw Data
            gnss_df, imu_df, gt_df = load_drive_data(
                drive_id, phone_name, gnss_rel, imu_rel, gt_rel
            )

            if gnss_df is None:
                # If GNSS file is missing but we are in test mode, we must preserve the rows.
                if split == "test":
                    print(
                        f"Warning: Missing GNSS for test trip {drive_id}-{phone_name}. Preserving rows with NaNs."
                    )
                    # Create a dataframe with index set to UnixTimeMillis and NaNs for features
                    merged = subset.set_index("UnixTimeMillis")
                    features_list.append(merged)
                continue

            # Process GNSS Features (Physics-based residuals)
            proc_gnss = process_gnss_data(gnss_df)

            # Aggregate Features (Sector-based)
            drive_feats = aggregate_features(proc_gnss, imu_df)

            # Align with Metadata Targets
            # Metadata uses 'UnixTimeMillis', aggregate_features index is 'utcTimeMillis'
            # We rename index in aggregate_features to match metadata for join
            drive_feats.index.name = "UnixTimeMillis"

            # Use variable join method (Cite debug_lesson_4)
            merged = align_timestamps(
                drive_feats, subset, on_col="UnixTimeMillis", how=join_method
            )

            if merged.empty:
                continue

            # Add WLS positions for target computation / reconstruction
            # WLS positions are in proc_gnss. We need to align them to the target timestamps.
            # Since WLS is per epoch, we take the first value for each timestamp.
            wls_pos = proc_gnss.groupby("utcTimeMillis")[
                [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ].first()
            wls_pos.index.name = "UnixTimeMillis"

            merged = merged.join(wls_pos, how=join_method)

            # Interpolate missing WLS positions for test set (Cite debug_lesson_4)
            if split == "test":
                wls_cols = [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
                # Sort by time to ensure interpolation makes sense
                merged = merged.sort_index()
                # Interpolate
                merged[wls_cols] = merged[wls_cols].interpolate(
                    method="linear", limit_direction="both"
                )

            # If Ground Truth file was loaded, merge AltitudeMeters if missing in metadata
            if gt_df is not None and "AltitudeMeters" in gt_df.columns:
                # GT might have slightly different timestamps or need alignment
                # Usually metadata is a subset of GT.
                # Let's map Altitude from GT to merged based on timestamp
                gt_alt = gt_df.set_index("UnixTimeMillis")["AltitudeMeters"]
                # Only keep rows present in merged
                merged["AltitudeMeters"] = gt_alt.reindex(merged.index).values

            # Compute Targets (ENU Residuals) and WLS LLA
            merged = compute_targets(merged)

            features_list.append(merged)

    if not features_list:
        raise ValueError(f"No data processed for {split} split!")

    full_df = pd.concat(features_list)

    # 2. Save to Cache
    full_df.to_parquet(cache_file)
    print(f"Saved {split} data to {cache_file}. Shape: {full_df.shape}")

    return full_df
