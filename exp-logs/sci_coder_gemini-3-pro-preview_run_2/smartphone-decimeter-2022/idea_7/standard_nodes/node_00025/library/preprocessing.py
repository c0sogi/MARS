import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import ecef_to_lla, deg_to_meters


def load_metadata(mode: str):
    """
    Load metadata for the specified mode (train, val, test).

    Args:
        mode: 'train', 'val', or 'test'

    Returns:
        DataFrame containing metadata.
    """
    if mode == "train":
        path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        path = Config.VAL_METADATA_PATH
    elif mode == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # In debug mode, sample a subset of trips
    if Config.DEBUG:
        unique_trips = df["tripId"].unique()
        if len(unique_trips) > Config.DEBUG_SAMPLE_SIZE:
            sampled_trips = np.random.choice(
                unique_trips, Config.DEBUG_SAMPLE_SIZE, replace=False
            )
            df = df[df["tripId"].isin(sampled_trips)].reset_index(drop=True)
            print(f"DEBUG: Sampled {len(sampled_trips)} trips for {mode}.")

    return df


def load_gnss_data(metadata_df: pd.DataFrame):
    """
    Load and aggregate raw GNSS data for trips in the metadata.

    Args:
        metadata_df: DataFrame containing trip metadata and file paths.

    Returns:
        DataFrame with aggregated GNSS features.
    """
    gnss_list = []

    # Get unique trips and their paths
    unique_trips = metadata_df[["tripId", "gnss_path"]].drop_duplicates()

    print(f"Loading GNSS data for {len(unique_trips)} trips...")

    for _, row in tqdm(unique_trips.iterrows(), total=len(unique_trips), disable=None):
        trip_id = row["tripId"]
        gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found for {trip_id} at {gnss_path}")
            continue

        try:
            # Read specific columns to save memory
            # Note: We need utcTimeMillis to aggregate
            cols_to_read = list(
                set(Config.RAW_GNSS_COLS) - {"tripId"}
            )  # tripId not in file usually
            df_trip = pd.read_csv(gnss_path, usecols=lambda c: c in cols_to_read)

            # Aggregation by epoch
            # We take the first WLS position as the baseline for that epoch
            # We count satellites and average signal metrics
            agg_funcs = {
                "Svid": "count",
                "Cn0DbHz": "mean",
                "RawPseudorangeUncertaintyMeters": "mean",
                "WlsPositionXEcefMeters": "first",
                "WlsPositionYEcefMeters": "first",
                "WlsPositionZEcefMeters": "first",
            }

            # Filter agg_funcs to only include columns present in the file
            agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_trip.columns}

            if "utcTimeMillis" not in df_trip.columns:
                print(f"Warning: utcTimeMillis missing in {gnss_path}")
                continue

            df_agg = df_trip.groupby("utcTimeMillis").agg(agg_funcs).reset_index()
            df_agg["tripId"] = trip_id

            # Rename columns to match expected features
            rename_map = {
                "Svid": "sat_count",
                "Cn0DbHz": "mean_cn0",
                "RawPseudorangeUncertaintyMeters": "mean_unc",
            }
            df_agg.rename(columns=rename_map, inplace=True)

            gnss_list.append(df_agg)

        except Exception as e:
            print(f"Error processing {trip_id}: {e}")

    if not gnss_list:
        return pd.DataFrame()

    return pd.concat(gnss_list, ignore_index=True)


def calculate_dynamics(df: pd.DataFrame):
    """
    Calculate velocity and other dynamic features from WLS positions.

    Args:
        df: DataFrame containing WLS ECEF coordinates.

    Returns:
        DataFrame with added dynamic features.
    """
    # Ensure sorted by time
    df = df.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

    # Convert WLS ECEF to LLA
    lat, lon, alt = ecef_to_lla(
        df["WlsPositionXEcefMeters"].values,
        df["WlsPositionYEcefMeters"].values,
        df["WlsPositionZEcefMeters"].values,
    )

    df["wls_lat"] = lat
    df["wls_lon"] = lon
    df["wls_alt"] = alt

    # Calculate velocities (first order difference)
    # We group by tripId to avoid diffing across trips

    # Helper to calculate diff in meters
    def get_diffs(group):
        # Time delta in seconds (approximate, usually 1s)
        dt = group["UnixTimeMillis"].diff() / 1000.0
        dt = dt.fillna(1.0)  # Avoid div by zero for first element

        # Lat/Lon diff in degrees
        d_lat = group["wls_lat"].diff().fillna(0.0)
        d_lon = group["wls_lon"].diff().fillna(0.0)
        d_alt = group["wls_alt"].diff().fillna(0.0)

        # Convert to meters
        # We use the group's mean latitude as reference for longitude scaling
        ref_lat = group["wls_lat"].mean()

        lat_m, lon_m = deg_to_meters(d_lat.values, d_lon.values, ref_lat)

        # Velocity
        group["vel_lat_m"] = lat_m / dt.values
        group["vel_lon_m"] = lon_m / dt.values
        group["vel_alt_m"] = d_alt / dt.values

        return group

    # Apply group-wise
    # Note: This can be slow. Optimized vectorization:
    # Since dataframe is sorted, we can just mask the first row of each trip

    # Global diff
    dt = df["UnixTimeMillis"].diff() / 1000.0

    # Handle dt=0 to avoid DivisionByZero/Inf
    dt = dt.replace(0.0, np.nan)

    d_lat = df["wls_lat"].diff()
    d_lon = df["wls_lon"].diff()
    d_alt = df["wls_alt"].diff()

    # Convert global diffs to meters
    # Use element-wise ref_lat (current row) for simplicity/vectorization
    lat_m, lon_m = deg_to_meters(d_lat.values, d_lon.values, df["wls_lat"].values)

    df["vel_lat_m"] = lat_m / dt
    df["vel_lon_m"] = lon_m / dt
    df["vel_alt_m"] = d_alt / dt

    # Replace Inf with NaN (caused by very small dt)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Mask boundaries
    # Where tripId changes, set velocity to 0
    mask = df["tripId"] != df["tripId"].shift(1)
    df.loc[mask, ["vel_lat_m", "vel_lon_m", "vel_alt_m"]] = 0.0

    # Fill NaNs (first row of entire DF)
    df[["vel_lat_m", "vel_lon_m", "vel_alt_m"]] = df[
        ["vel_lat_m", "vel_lon_m", "vel_alt_m"]
    ].fillna(0.0)

    return df


def align_data(metadata_df: pd.DataFrame, gnss_df: pd.DataFrame, mode: str):
    """
    Merge metadata (Ground Truth/Submission) with GNSS data.
    Computes targets for train/val.
    """
    # Rename utcTimeMillis to UnixTimeMillis for merging
    gnss_df = gnss_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

    # Merge
    # For test, we use left join to ensure we have all rows required for submission
    # For train/val, we use inner join to ensure we have valid features
    how_merge = "left" if mode == "test" else "inner"

    merged_df = pd.merge(
        metadata_df, gnss_df, on=["tripId", "UnixTimeMillis"], how=how_merge
    )

    # Handle missing GNSS data (mostly for test set gaps)
    # Forward fill within trip, then backward fill, then fill 0
    # Apply to all modes to avoid NaNs in training
    cols_to_fill = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "sat_count",
        "mean_cn0",
        "mean_unc",
    ]
    # Only fill columns that exist
    cols_to_fill = [c for c in cols_to_fill if c in merged_df.columns]

    # Sort for filling
    merged_df = merged_df.sort_values(["tripId", "UnixTimeMillis"])
    merged_df[cols_to_fill] = merged_df.groupby("tripId")[cols_to_fill].ffill().bfill()
    merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(0)

    # Calculate Dynamics (Velocities) and WLS LLA
    merged_df = calculate_dynamics(merged_df)

    # Calculate Targets for Train/Val
    if mode in ["train", "val"]:
        # Target is GT - WLS in meters
        d_lat = merged_df["LatitudeDegrees"] - merged_df["wls_lat"]
        d_lon = merged_df["LongitudeDegrees"] - merged_df["wls_lon"]

        t_lat_m, t_lon_m = deg_to_meters(
            d_lat.values, d_lon.values, merged_df["wls_lat"].values
        )

        merged_df["target_lat_m"] = t_lat_m
        merged_df["target_lon_m"] = t_lon_m

    return merged_df


def process_dataset(mode: str, load_cached_data: bool = True):
    """
    Main processing function. Loads, merges, and engineers features.

    Args:
        mode: 'train', 'val', or 'test'
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        Processed DataFrame ready for dataset creation.
    """
    # Determine cache path
    if mode == "train":
        cache_path = Config.TRAIN_CACHE_PATH
    elif mode == "val":
        cache_path = Config.VAL_CACHE_PATH
    elif mode == "test":
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            print(f"Successfully loaded {len(df)} rows.")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {mode} data...")

    # 1. Load Metadata
    meta_df = load_metadata(mode)

    # 2. Load GNSS
    gnss_df = load_gnss_data(meta_df)

    if gnss_df.empty:
        raise ValueError("No GNSS data loaded. Check input files.")

    # 3. Align and Feature Engineering
    df = align_data(meta_df, gnss_df, mode)

    # 4. Save to cache
    print(f"Saving {mode} data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    print(f"Processed {len(df)} rows for {mode}.")
    return df
