import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import ecef_to_geodetic, geodetic_to_enu


def load_and_aggregate_gnss(gnss_path):
    """
    Loads raw GNSS data, aggregates it to 1Hz, and computes baseline positions.

    Args:
        gnss_path (str): Relative path to the device_gnss.csv file.

    Returns:
        pd.DataFrame: Aggregated GNSS data with 1Hz timestamps and baseline coordinates.
    """
    # Define columns to read: features defined in Config + WLS positions for baseline
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    # Use set to avoid duplicates if Config already has them
    cols_to_read = list(set(Config.GNSS_COLS + wls_cols))

    full_path = os.path.join(Config.INPUT_DIR, gnss_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"GNSS file not found: {full_path}")

    # Load data
    df = pd.read_csv(full_path, usecols=lambda c: c in cols_to_read)

    # 1Hz Alignment: Round utcTimeMillis to nearest second (1000ms)
    # This aligns the high-frequency raw data with the 1Hz ground truth
    df["UnixTimeMillis"] = (np.round(df["utcTimeMillis"] / 1000) * 1000).astype(
        np.int64
    )

    # Prepare Aggregation Dictionary
    agg_dict = {}

    # Add feature aggregations from Config
    for col, funcs in Config.GNSS_AGG_CONFIG.items():
        if col in df.columns:
            agg_dict[col] = funcs

    # Add WLS position aggregation (mean of the epoch)
    for col in wls_cols:
        if col in df.columns:
            agg_dict[col] = "mean"

    # Perform GroupBy Aggregation
    df_agg = df.groupby("UnixTimeMillis").agg(agg_dict)

    # Flatten MultiIndex columns (e.g., ('Cn0DbHz', 'mean') -> 'Cn0DbHz_mean')
    new_columns = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            new_col_name = f"{col[0]}_{col[1]}"
        else:
            new_col_name = col
        new_columns.append(new_col_name)
    df_agg.columns = new_columns

    df_agg = df_agg.reset_index()

    # Compute Baseline Geodetic Coordinates from averaged WLS ECEF
    wls_x_col = "WlsPositionXEcefMeters_mean"
    wls_y_col = "WlsPositionYEcefMeters_mean"
    wls_z_col = "WlsPositionZEcefMeters_mean"

    if (
        wls_x_col in df_agg.columns
        and wls_y_col in df_agg.columns
        and wls_z_col in df_agg.columns
    ):
        # Vectorized conversion
        x = df_agg[wls_x_col].values
        y = df_agg[wls_y_col].values
        z = df_agg[wls_z_col].values

        lat, lon, alt = ecef_to_geodetic(x, y, z)

        df_agg["BaselineLat"] = lat
        df_agg["BaselineLon"] = lon
        df_agg["BaselineAlt"] = alt
    else:
        # Fallback if WLS data is missing (should not happen in this dataset)
        df_agg["BaselineLat"] = np.nan
        df_agg["BaselineLon"] = np.nan
        df_agg["BaselineAlt"] = np.nan

    return df_agg


def calculate_residuals(df):
    """
    Calculates ENU residuals (North, East) in meters between Ground Truth and Baseline.

    Args:
        df (pd.DataFrame): DataFrame containing GT and Baseline coordinates.

    Returns:
        pd.DataFrame: DataFrame with added 'lat_res_m' and 'lon_res_m' columns.
    """
    # Ensure required columns exist
    req_cols = ["LatitudeDegrees", "LongitudeDegrees", "BaselineLat", "BaselineLon"]
    if not all(col in df.columns for col in req_cols):
        # If GT is missing (e.g. test set), we cannot calculate residuals
        return df

    # Calculate residuals using the utility function
    north, east = geodetic_to_enu(
        df["LatitudeDegrees"].values,
        df["LongitudeDegrees"].values,
        df["BaselineLat"].values,
        df["BaselineLon"].values,
    )

    df["lat_res_m"] = north
    df["lon_res_m"] = east

    return df


def process_dataset(metadata_path, load_cached_data=True, split_name="train"):
    """
    Main data processing function.
    Loads metadata, processes each drive (aggregating GNSS, merging GT),
    caches results, and returns a combined DataFrame.

    Args:
        metadata_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.
        split_name (str): Name of the split (train/val/test) for cache naming.

    Returns:
        pd.DataFrame: Combined processed dataset.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Identify unique drive-phone pairs to process
    pairs = meta_df[["drive_id", "phone_name"]].drop_duplicates().values

    processed_dfs = []

    print(f"Processing {len(pairs)} drive-phone pairs for split: {split_name}")

    for drive_id, phone_name in pairs:
        # Construct cache filename
        cache_file = os.path.join(
            Config.WORKING_DIR, f"{drive_id}_{phone_name}_{split_name}.parquet"
        )

        loaded = False
        if load_cached_data and os.path.exists(cache_file):
            try:
                df_pair = pd.read_parquet(cache_file)
                loaded = True
            except Exception as e:
                print(f"Failed to load cache {cache_file}: {e}")

        if not loaded:
            # 1. Get metadata subset for this pair (contains GT or query timestamps)
            pair_meta = meta_df[
                (meta_df["drive_id"] == drive_id)
                & (meta_df["phone_name"] == phone_name)
            ].copy()

            if pair_meta.empty:
                continue

            # 2. Load and Aggregate GNSS
            # gnss_path is in the metadata. All rows for this pair have same gnss_path.
            gnss_rel_path = pair_meta.iloc[0]["gnss_path"]

            try:
                gnss_agg = load_and_aggregate_gnss(gnss_rel_path)
            except Exception as e:
                print(f"Error processing GNSS for {drive_id} {phone_name}: {e}")
                continue

            # 3. Merge with Metadata
            # Create a rounded timestamp for alignment in metadata
            pair_meta["UnixTimeMillis_rounded"] = (
                np.round(pair_meta["UnixTimeMillis"] / 1000) * 1000
            ).astype(np.int64)

            # Rename aggregated GNSS timestamp to match
            gnss_agg = gnss_agg.rename(
                columns={"UnixTimeMillis": "UnixTimeMillis_rounded"}
            )

            # Inner join on UnixTimeMillis_rounded to align timestamps
            df_pair = pd.merge(
                pair_meta, gnss_agg, on="UnixTimeMillis_rounded", how="inner"
            )

            # 4. Calculate Residuals (only if GT exists)
            if (
                "LatitudeDegrees" in df_pair.columns
                and "LongitudeDegrees" in df_pair.columns
            ):
                df_pair = calculate_residuals(df_pair)

            # 5. Save to Cache
            try:
                df_pair.to_parquet(cache_file, index=False)
            except Exception as e:
                print(f"Failed to save cache {cache_file}: {e}")

        processed_dfs.append(df_pair)

    if not processed_dfs:
        print("No data processed.")
        return pd.DataFrame()

    final_df = pd.concat(processed_dfs, ignore_index=True)
    return final_df
