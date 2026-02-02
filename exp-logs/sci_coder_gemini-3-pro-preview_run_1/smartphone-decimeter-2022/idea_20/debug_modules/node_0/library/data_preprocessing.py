import os
import numpy as np
import pandas as pd
from library.config import CFG
from library.utils import wgs84_to_enu


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.

    Args:
        x, y, z: ECEF coordinates in meters (numpy arrays).

    Returns:
        lat, lon: Latitude and Longitude in degrees.
    """
    # WGS84 Ellipsoid constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    return np.degrees(lat), np.degrees(lon)


def aggregate_gnss_features(gnss_df):
    """
    Aggregates raw GNSS measurements into 1Hz statistics.

    Args:
        gnss_df: DataFrame containing raw GNSS measurements.

    Returns:
        DataFrame with aggregated features indexed by UnixTimeMillis.
    """
    # Create a copy to avoid modifying original
    df = gnss_df.copy()

    # Round timestamp to nearest second for alignment
    df["UnixTimeMillis"] = np.round(df["utcTimeMillis"] / 1000) * 1000
    df["UnixTimeMillis"] = df["UnixTimeMillis"].astype(np.int64)

    # Feature Engineering
    # Azimuth: Decompose into sin/cos components to handle cyclic nature
    df["SvAzimuth_rad"] = np.deg2rad(df["SvAzimuthDegrees"].fillna(0))
    df["SvAzimuth_sin"] = np.sin(df["SvAzimuth_rad"])
    df["SvAzimuth_cos"] = np.cos(df["SvAzimuth_rad"])

    # Handle missing values in critical columns before aggregation
    df["Cn0DbHz"] = df["Cn0DbHz"].fillna(0)
    df["SvElevationDegrees"] = df["SvElevationDegrees"].fillna(0)
    df["PseudorangeRateMetersPerSecond"] = df["PseudorangeRateMetersPerSecond"].fillna(
        0
    )
    df["RawPseudorangeUncertaintyMeters"] = df[
        "RawPseudorangeUncertaintyMeters"
    ].fillna(0)

    # Define aggregations
    aggs = {
        "Cn0DbHz": ["mean", "std", "min", "max"],
        "SvElevationDegrees": ["mean", "std", "min", "max"],
        "SvAzimuth_sin": ["mean"],
        "SvAzimuth_cos": ["mean"],
        "PseudorangeRateMetersPerSecond": ["mean", "std"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
        "Svid": ["count"],  # SatCount
    }

    # Group by timestamp
    grouped = df.groupby("UnixTimeMillis").agg(aggs)

    # Flatten MultiIndex columns
    grouped.columns = [
        f"{col[0]}_{col[1]}" if col[0] != "Svid" else "SatCount"
        for col in grouped.columns
    ]

    # Rename specific columns to match CFG.FEATURE_COLS
    rename_map = {
        "SvAzimuth_sin_mean": "SvAzimuthDegrees_sin_mean",
        "SvAzimuth_cos_mean": "SvAzimuthDegrees_cos_mean",
    }
    grouped = grouped.rename(columns=rename_map)

    # Fill NaNs resulting from aggregation (e.g., std of single value is NaN)
    grouped = grouped.fillna(0)

    return grouped.reset_index()


def process_dataset(metadata_path, cache_path, load_cached_data=True, debug=False):
    """
    Main data processing function. Loads metadata, processes raw GNSS files,
    aligns data, computes targets, and manages caching.

    Args:
        metadata_path: Path to the metadata CSV file.
        cache_path: Path where the processed Parquet file should be saved/loaded.
        load_cached_data: Boolean, whether to attempt loading from cache.
        debug: Boolean, if True, processes a small subset of data.

    Returns:
        DataFrame containing processed features and targets (if available).
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing data from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if debug:
        # Sample a few drives for debugging
        unique_drives = df_meta["drive_id"].unique()
        sample_drives = unique_drives[:2] if len(unique_drives) >= 2 else unique_drives
        df_meta = df_meta[df_meta["drive_id"].isin(sample_drives)].copy()
        print(
            f"Debug mode: Processing {len(df_meta)} rows from drives: {sample_drives}"
        )

    # Get unique drive-phone combinations to process raw files efficiently
    unique_drives = df_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    processed_dfs = []

    for _, row in unique_drives.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_rel_path = row["gnss_path"]

        gnss_full_path = os.path.join(CFG.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_full_path):
            # print(f"Warning: File not found {gnss_full_path}")
            continue

        # Load Raw Data
        try:
            gnss_df = pd.read_csv(gnss_full_path)
        except Exception as e:
            print(f"Error reading {gnss_full_path}: {e}")
            continue

        # 1. Aggregate Features
        features_df = aggregate_gnss_features(gnss_df)

        # 2. Get Baseline WLS (convert ECEF to LLA)
        # We take the first WLS position for each second
        wls_cols = [
            "utcTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        # Check if columns exist (some files might be malformed)
        if not all(col in gnss_df.columns for col in wls_cols):
            print(f"Missing WLS columns in {gnss_rel_path}")
            continue

        wls_df = gnss_df[wls_cols].copy()
        wls_df["UnixTimeMillis"] = np.round(wls_df["utcTimeMillis"] / 1000) * 1000
        wls_df["UnixTimeMillis"] = wls_df["UnixTimeMillis"].astype(np.int64)

        # Drop duplicates to get one position per second
        wls_df = wls_df.drop_duplicates(subset=["UnixTimeMillis"])

        # Convert ECEF to Lat/Lon
        lat, lon = ecef_to_lla(
            wls_df["WlsPositionXEcefMeters"].values,
            wls_df["WlsPositionYEcefMeters"].values,
            wls_df["WlsPositionZEcefMeters"].values,
        )
        wls_df["WlsLatitudeDegrees"] = lat
        wls_df["WlsLongitudeDegrees"] = lon

        # Merge Features with Baseline
        drive_data = pd.merge(
            features_df,
            wls_df[["UnixTimeMillis", "WlsLatitudeDegrees", "WlsLongitudeDegrees"]],
            on="UnixTimeMillis",
            how="inner",
        )

        # 3. Merge with Metadata (Ground Truth or Test Targets)
        # Filter metadata for this drive
        drive_meta_subset = df_meta[
            (df_meta["drive_id"] == drive_id) & (df_meta["phone_name"] == phone_name)
        ]

        if drive_meta_subset.empty:
            continue

        # Inner join to ensure we only keep timestamps present in both (and required by meta)
        # This aligns the 1Hz raw features with the 1Hz ground truth/submission requirements
        merged = pd.merge(
            drive_meta_subset, drive_data, on="UnixTimeMillis", how="inner"
        )

        # 4. Calculate Targets (if training data)
        if "LatitudeDegrees" in merged.columns and "LongitudeDegrees" in merged.columns:
            # Calculate offsets relative to WLS baseline
            # Target = GT - Baseline (in meters)
            # We use wgs84_to_enu with the Baseline as the reference

            east, north = wgs84_to_enu(
                merged["LatitudeDegrees"].values,
                merged["LongitudeDegrees"].values,
                merged["WlsLatitudeDegrees"].values,
                merged["WlsLongitudeDegrees"].values,
            )

            merged["target_east"] = east
            merged["target_north"] = north

        processed_dfs.append(merged)

    if not processed_dfs:
        print("No data processed!")
        return pd.DataFrame()

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    print(f"Saved processed data to {cache_path}")

    return full_df
