import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEG_TO_M_LAT,
    DEG_TO_M_LON,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import ecef_to_lla, degrees_to_meters


def load_metadata(split: str) -> pd.DataFrame:
    """
    Load metadata for the specified split.

    Args:
        split: 'train', 'val', or 'test'.

    Returns:
        DataFrame containing metadata.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    return df


def process_gnss_data(gnss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw GNSS data by epoch (utcTimeMillis) to create feature columns.

    Args:
        gnss_df: Raw GNSS DataFrame.

    Returns:
        Aggregated DataFrame with one row per epoch.
    """
    # Define aggregation dictionary
    agg_funcs = {
        "Svid": "count",
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "SvElevationDegrees": ["mean", "std", "min"],
        "SvAzimuthDegrees": ["mean", "std"],
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Filter for columns that actually exist in the dataframe
    existing_cols = set(gnss_df.columns)
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in existing_cols}

    # Group by time
    df_agg = gnss_df.groupby("utcTimeMillis").agg(agg_funcs)

    # Flatten MultiIndex columns
    df_agg.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_agg.columns.values
    ]

    # Rename columns to match config expectations
    rename_map = {
        "Svid_count": "sv_count",
        "Cn0DbHz_mean": "mean_cn0",
        "RawPseudorangeUncertaintyMeters_mean": "mean_uncertainty",
        "SvElevationDegrees_mean": "mean_sv_elevation",
        "SvElevationDegrees_std": "std_sv_elevation",
        "SvElevationDegrees_min": "min_sv_elevation",
        "SvAzimuthDegrees_mean": "mean_sv_azimuth",
        "SvAzimuthDegrees_std": "std_sv_azimuth",
        "WlsPositionXEcefMeters_first": "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters_first": "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters_first": "WlsPositionZEcefMeters",
    }

    # Handle cases where columns might be missing (e.g. if SvElevationDegrees wasn't in input)
    final_rename = {k: v for k, v in rename_map.items() if k in df_agg.columns}
    df_agg = df_agg.rename(columns=final_rename)

    # Reset index to make utcTimeMillis a column
    df_agg = df_agg.reset_index()

    # Fill NaN values resulting from std of single samples
    df_agg = df_agg.fillna(0)

    return df_agg


def process_trip(
    trip_id: str, trip_meta_df: pd.DataFrame, is_test: bool
) -> pd.DataFrame:
    """
    Load GNSS data for a trip, merge with metadata (ground truth), and calculate features.

    Args:
        trip_id: Unique trip identifier.
        trip_meta_df: Metadata rows corresponding to this trip.
        is_test: Boolean indicating if this is test data (no ground truth targets needed, but timestamps must align).

    Returns:
        DataFrame containing features and targets (if not test) for the trip.
    """
    # Get file path from the first row of metadata for this trip
    # Note: We assume all rows for a trip point to the same GNSS file
    gnss_rel_path = trip_meta_df.iloc[0]["gnss_path"]
    gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)

    if not os.path.exists(gnss_path):
        print(f"Warning: GNSS file not found for trip {trip_id}: {gnss_path}")
        return pd.DataFrame()

    # Load Raw GNSS
    gnss_df = pd.read_csv(gnss_path)

    # Aggregate GNSS
    gnss_agg = process_gnss_data(gnss_df)

    # Merge with Metadata
    # Metadata contains the target timestamps (UnixTimeMillis)
    # GNSS contains utcTimeMillis. We assume they are aligned.
    merged = pd.merge(
        trip_meta_df,
        gnss_agg,
        left_on="UnixTimeMillis",
        right_on="utcTimeMillis",
        how="inner",  # We only care about epochs where we have both requirements (GT/Submission) and Data
    )

    if merged.empty:
        return pd.DataFrame()

    # Sanitize: Drop rows where WLS is 0 (filled from NaN)
    # This prevents massive residuals from (0,0,0) ECEF
    valid_mask = (
        (merged["WlsPositionXEcefMeters"] != 0)
        & (merged["WlsPositionYEcefMeters"] != 0)
        & (merged["WlsPositionZEcefMeters"] != 0)
    )
    merged = merged[valid_mask].copy()

    if merged.empty:
        return pd.DataFrame()

    # Calculate WLS LLA
    wls_lat, wls_lon, wls_alt = ecef_to_lla(
        merged["WlsPositionXEcefMeters"].values,
        merged["WlsPositionYEcefMeters"].values,
        merged["WlsPositionZEcefMeters"].values,
    )

    merged["wls_lat"] = wls_lat
    merged["wls_lon"] = wls_lon
    merged["wls_alt"] = wls_alt

    # Calculate Velocities (First order differences)
    # We calculate these on the full sequence before any windowing
    # Fill NaNs with 0 for the first element
    merged["vel_lat_m"] = merged["wls_lat"].diff().fillna(0) * DEG_TO_M_LAT
    merged["vel_lon_m"] = (
        merged["wls_lon"].diff().fillna(0)
        * DEG_TO_M_LON
        * np.cos(np.radians(merged["wls_lat"]))
    )
    merged["vel_alt_m"] = merged["wls_alt"].diff().fillna(0)

    # Calculate Targets (Residuals) if not test
    if not is_test:
        # Ground Truth is in LatitudeDegrees, LongitudeDegrees
        # Calculate residuals in meters
        res_lat_m, res_lon_m = degrees_to_meters(
            merged["LatitudeDegrees"].values - merged["wls_lat"].values,
            merged["LongitudeDegrees"].values - merged["wls_lon"].values,
            merged["wls_lat"].values,
        )
        merged["res_lat_m"] = res_lat_m
        merged["res_lon_m"] = res_lon_m

    return merged


def get_dataset(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main function to get the dataset for a specific split.
    Handles caching and processing.

    Args:
        split: 'train', 'val', or 'test'.
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        Processed DataFrame ready for dataset creation.
    """
    cache_path = os.path.join(WORKING_DIR, f"{split}_data.parquet")

    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {split} data...")
    meta_df = load_metadata(split)

    # Debug mode: sample trips
    if DEBUG:
        unique_trips = meta_df["tripId"].unique()
        if len(unique_trips) > DEBUG_SAMPLE_SIZE:
            sample_trips = np.random.choice(
                unique_trips, DEBUG_SAMPLE_SIZE, replace=False
            )
            meta_df = meta_df[meta_df["tripId"].isin(sample_trips)].copy()
            print(f"Debug mode: Sampled {DEBUG_SAMPLE_SIZE} trips.")

    processed_trips = []
    unique_trips = meta_df["tripId"].unique()

    for trip_id in unique_trips:
        trip_meta = meta_df[meta_df["tripId"] == trip_id].sort_values("UnixTimeMillis")
        trip_data = process_trip(trip_id, trip_meta, is_test=(split == "test"))

        if not trip_data.empty:
            processed_trips.append(trip_data)

    if not processed_trips:
        raise ValueError(f"No data processed for split {split}")

    full_df = pd.concat(processed_trips, ignore_index=True)

    # 3. Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    full_df.to_parquet(cache_path, index=False)

    return full_df
