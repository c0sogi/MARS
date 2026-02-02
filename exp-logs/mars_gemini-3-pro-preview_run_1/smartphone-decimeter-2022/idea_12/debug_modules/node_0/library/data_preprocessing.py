import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from library.config import Config
from library.utils import WGS84_to_Meters


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.
    Vectorized implementation using Ferrari's method or similar closed-form approximation.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = 2 * f - f**2
    ep2 = (a**2 - b**2) / b**2

    r = np.sqrt(x**2 + y**2)
    E2 = a**2 - b**2
    F = 54 * b**2 * z**2
    G = r**2 + (1 - e2) * z**2 - e2 * E2
    C = (e2**2 * F * r**2) / (G**3)
    S = (1 + C + np.sqrt(C**2 + 2 * C)) ** (1 / 3)
    P = F / (3 * (S + 1 / S + 1) ** 2 * G**2)
    Q = np.sqrt(1 + 2 * e2**2 * P)
    r0 = -(P * e2 * r) / (1 + Q) + np.sqrt(
        (a**2 / 2) * (1 + 1 / Q) - (P * (1 - e2) * z**2) / (Q * (1 + Q)) - P * r**2 / 2
    )
    U = np.sqrt((r - e2 * r0) ** 2 + z**2)
    V = np.sqrt((r - e2 * r0) ** 2 + (1 - e2) * z**2)
    z0 = (b**2 * z) / (a * V)

    alt = U * (1 - b**2 / (a * V))
    lat = np.arctan((z + ep2 * z0) / r)
    lon = np.arctan2(y, x)

    return np.degrees(lat), np.degrees(lon), alt


def load_and_aggregate_gnss(gnss_path):
    """
    Loads raw GNSS data, rounds timestamps, aggregates features, and extracts baseline WLS position.

    Args:
        gnss_path (str): Relative path to the GNSS csv file.

    Returns:
        pd.DataFrame: Aggregated GNSS features indexed by 'UnixTimeMillis'.
    """
    full_path = os.path.join(Config.INPUT_DIR, gnss_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"GNSS file not found: {full_path}")

    # Read only necessary columns to save memory
    cols_to_read = list(
        set(
            Config.RAW_GNSS_COLS
            + [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        )
    )
    df = pd.read_csv(full_path, usecols=cols_to_read)

    # Temporal Quantization: Round to nearest second (1000 ms)
    # This aligns the high-frequency GNSS logs (or slightly offset logs) to the 1Hz ground truth
    df["UnixTimeMillis"] = (np.round(df["utcTimeMillis"] / 1000.0) * 1000).astype(
        np.int64
    )

    # Group by the quantized timestamp
    grouped = df.groupby("UnixTimeMillis")

    # 1. Feature Aggregation
    agg_dict = Config.FEATURE_AGGREGATIONS.copy()

    # Perform aggregation
    df_agg = grouped.agg(agg_dict)

    # Flatten MultiIndex columns (e.g., ('Cn0DbHz', 'mean') -> 'Cn0DbHz_mean')
    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]

    # 2. Metadata Features (SatCount)
    if Config.INCLUDE_SAT_COUNT:
        df_agg["SatCount"] = grouped.size()

    # 3. Extract Baseline Position (WLS)
    # We take the first value per group since WLS position is estimated per epoch
    # and repeated for each satellite row in that epoch.
    wls_pos = grouped[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].first()

    # Convert ECEF to LLA
    lat_wls, lon_wls, _ = ecef_to_lla(
        wls_pos["WlsPositionXEcefMeters"].values,
        wls_pos["WlsPositionYEcefMeters"].values,
        wls_pos["WlsPositionZEcefMeters"].values,
    )

    df_agg["wls_lat"] = lat_wls
    df_agg["wls_lon"] = lon_wls

    # Reset index to make UnixTimeMillis a column
    df_agg = df_agg.reset_index()

    return df_agg


def process_dataset(metadata_path, dataset_type="train", debug_size=None):
    """
    Generic function to process a dataset defined by a metadata CSV.

    Args:
        metadata_path (str): Path to the metadata CSV.
        dataset_type (str): 'train' or 'test'.
        debug_size (int, optional): Limit number of drives for debugging.

    Returns:
        pd.DataFrame: Processed dataset with features and (if train) targets.
    """
    df_meta = pd.read_csv(metadata_path)

    # Filter unique drives to iterate over
    unique_drives = df_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    if debug_size is not None:
        unique_drives = unique_drives.head(debug_size)

    processed_frames = []

    print(f"Processing {dataset_type} dataset: {len(unique_drives)} drives...")

    for _, row in unique_drives.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        try:
            # 1. Load and Aggregate Raw GNSS
            df_gnss = load_and_aggregate_gnss(gnss_path)

            # 2. Filter Metadata for this drive
            # We need to align the aggregated GNSS with the specific timestamps requested in metadata
            drive_meta = df_meta[
                (df_meta["drive_id"] == drive_id)
                & (df_meta["phone_name"] == phone_name)
            ].copy()

            # Ensure metadata timestamps are also quantized if they aren't already perfect integers
            # (Though usually GT/Submission timestamps are clean integers)
            drive_meta["UnixTimeMillis"] = (
                np.round(drive_meta["UnixTimeMillis"] / 1000.0) * 1000
            ).astype(np.int64)

            # 3. Merge
            # Inner join ensures we only keep epochs where we have both requirements (GT/Sub) and Data
            merged = pd.merge(drive_meta, df_gnss, on="UnixTimeMillis", how="inner")

            # 4. Calculate Targets (Training Only)
            if dataset_type == "train":
                # Calculate offsets: Target (GT) - Baseline (WLS)
                # Note: WGS84_to_Meters returns (delta_north, delta_east)
                d_north, d_east = WGS84_to_Meters(
                    merged["wls_lat"].values,
                    merged["wls_lon"].values,
                    merged["LatitudeDegrees"].values,
                    merged["LongitudeDegrees"].values,
                )
                merged["delta_north"] = d_north
                merged["delta_east"] = d_east

            # Keep track of drive/phone for splitting later
            # (Already in merged from drive_meta)

            processed_frames.append(merged)

        except Exception as e:
            print(f"Error processing drive {drive_id} {phone_name}: {e}")
            continue

    if not processed_frames:
        return pd.DataFrame()

    return pd.concat(processed_frames, ignore_index=True)


def prepare_training_data(load_cached_data=True):
    """
    Prepares the training and validation data.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df)
    """
    cache_path = os.path.join(Config.WORKING_DIR, "train_val_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training data from {cache_path}")
        full_df = pd.read_parquet(cache_path)
    else:
        print("Computing training data from scratch...")
        # We use the train_metadata.csv which contains the full training set (before internal split)
        # Note: The Config.TRAIN_METADATA_PATH points to the file generated by generate_metadata()
        # which is actually ~80% of the total data. The val_metadata.csv has the other 20%.
        # To be robust, we should process both and combine, or just process them separately.
        # However, the standard pipeline usually expects us to handle the split here or load pre-split metadata.
        # Given the generate_metadata script splits by drive, we can process them independently.

        df_train_part = process_dataset(
            Config.TRAIN_METADATA_PATH,
            dataset_type="train",
            debug_size=Config.DEBUG_SAMPLE_SIZE,
        )
        df_val_part = process_dataset(
            Config.VAL_METADATA_PATH,
            dataset_type="train",
            debug_size=Config.DEBUG_SAMPLE_SIZE,
        )

        full_df = pd.concat([df_train_part, df_val_part], ignore_index=True)

        # Save to cache
        print(f"Saving processed data to {cache_path}")
        full_df.to_parquet(cache_path, index=False)

    # Perform Train/Val Split
    # We use GroupShuffleSplit on drive_id to ensure no leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    train_inds, val_inds = next(splitter.split(full_df, groups=full_df["drive_id"]))

    train_df = full_df.iloc[train_inds].reset_index(drop=True)
    val_df = full_df.iloc[val_inds].reset_index(drop=True)

    print(f"Data Split: Train {len(train_df)} samples, Val {len(val_df)} samples")

    return train_df, val_df


def prepare_test_data(load_cached_data=True):
    """
    Prepares the test data for inference.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Processed test dataframe ready for inference.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test data from {cache_path}")
        test_df = pd.read_parquet(cache_path)
    else:
        print("Computing test data from scratch...")
        test_df = process_dataset(
            Config.TEST_METADATA_PATH,
            dataset_type="test",
            debug_size=Config.DEBUG_SAMPLE_SIZE,
        )

        print(f"Saving processed test data to {cache_path}")
        test_df.to_parquet(cache_path, index=False)

    print(f"Test Data: {len(test_df)} samples")
    return test_df
