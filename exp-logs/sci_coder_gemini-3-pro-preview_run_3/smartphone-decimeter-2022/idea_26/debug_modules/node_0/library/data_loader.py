import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)


def load_metadata(split_name: str):
    """
    Load the metadata CSV for a specific split.

    Parameters:
    -----------
    split_name : str
        One of 'train', 'val', 'test'.

    Returns:
    --------
    pd.DataFrame
        The metadata dataframe.
    """
    if split_name == "train":
        path = TRAIN_METADATA_PATH
    elif split_name == "val":
        path = VAL_METADATA_PATH
    elif split_name == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split_name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def load_gnss_dataframe(
    metadata_df: pd.DataFrame, split_name: str, load_cached_data: bool = True
):
    """
    Load raw GNSS data for all drives in the metadata.
    Performs basic cleaning (dropping NaNs in critical columns).
    Caches the result as a parquet file.

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe containing 'gnss_path', 'drive_id', 'phone_name'.
    split_name : str
        Name of the split (used for cache filename).
    load_cached_data : bool
        If True, attempts to load from cache first.

    Returns:
    --------
    pd.DataFrame
        Combined GNSS dataframe for the split.
    """
    cache_path = os.path.join(WORKING_DIR, f"gnss_{split_name}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading GNSS data for '{split_name}' from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Loading GNSS data for '{split_name}' from source CSVs...")

    # Identify unique GNSS files to load
    # We group by tripId or just path. Metadata has one row per timestamp,
    # but multiple rows share the same gnss_path.
    unique_files = metadata_df[
        ["gnss_path", "drive_id", "phone_name"]
    ].drop_duplicates()

    gnss_list = []

    # Columns to load (optimization)
    # We load columns needed for WLS, residuals, and state features
    use_cols = [
        "utcTimeMillis",
        "TimeNanos",
        "FullBiasNanos",
        "BiasNanos",
        "BiasUncertaintyNanos",
        "DriftNanosPerSecond",
        "DriftUncertaintyNanosPerSecond",
        "Svid",
        "SignalType",
        "Cn0DbHz",
        "ConstellationType",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
        "RawPseudorangeMeters",
        "RawPseudorangeUncertaintyMeters",
        "AccumulatedDeltaRangeMeters",
        "AccumulatedDeltaRangeState",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "IsrbMeters",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
    ]

    for _, row in unique_files.iterrows():
        rel_path = row["gnss_path"]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: GNSS file not found: {full_path}")
            continue

        try:
            # Load only necessary columns if they exist
            # First read header to check existence
            header = pd.read_csv(full_path, nrows=0).columns.tolist()
            cols_to_read = [c for c in use_cols if c in header]

            df = pd.read_csv(full_path, usecols=cols_to_read)

            # Basic Cleaning
            # Drop rows where Pseudorange is missing (cannot position without it)
            if "RawPseudorangeMeters" in df.columns:
                df = df.dropna(subset=["RawPseudorangeMeters"])

            # Add identifiers
            df["drive_id"] = drive_id
            df["phone_name"] = phone_name
            # Construct tripId for joining
            df["tripId"] = drive_id + "-" + phone_name

            gnss_list.append(df)

        except Exception as e:
            print(f"Error reading {full_path}: {e}")

    if not gnss_list:
        raise ValueError(f"No GNSS data loaded for split {split_name}")

    combined_df = pd.concat(gnss_list, ignore_index=True)

    # 3. Save to cache
    print(f"Saving GNSS data for '{split_name}' to cache: {cache_path}")
    combined_df.to_parquet(cache_path, index=False)

    return combined_df


def load_ground_truth(
    metadata_df: pd.DataFrame, split_name: str, load_cached_data: bool = True
):
    """
    Load ground truth data for the drives in the metadata.
    Only applicable for 'train' and 'val' splits.

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe.
    split_name : str
        Name of the split.
    load_cached_data : bool
        If True, attempts to load from cache.

    Returns:
    --------
    pd.DataFrame
        Combined Ground Truth dataframe.
    """
    if split_name == "test":
        print("Ground truth not available for test split.")
        return pd.DataFrame()

    cache_path = os.path.join(WORKING_DIR, f"gt_{split_name}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading GT data for '{split_name}' from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Loading GT data for '{split_name}' from source CSVs...")

    # Identify unique GT files to load
    # Note: gt_path column must exist in metadata for train/val
    if "gt_path" not in metadata_df.columns:
        raise ValueError("Metadata does not contain 'gt_path' column.")

    unique_files = metadata_df[["gt_path", "drive_id", "phone_name"]].drop_duplicates()

    gt_list = []

    for _, row in unique_files.iterrows():
        rel_path = row["gt_path"]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: GT file not found: {full_path}")
            continue

        try:
            df = pd.read_csv(full_path)

            # Add identifiers
            df["drive_id"] = drive_id
            df["phone_name"] = phone_name
            df["tripId"] = drive_id + "-" + phone_name

            gt_list.append(df)

        except Exception as e:
            print(f"Error reading {full_path}: {e}")

    if not gt_list:
        raise ValueError(f"No GT data loaded for split {split_name}")

    combined_df = pd.concat(gt_list, ignore_index=True)

    # 3. Save to cache
    print(f"Saving GT data for '{split_name}' to cache: {cache_path}")
    combined_df.to_parquet(cache_path, index=False)

    return combined_df
