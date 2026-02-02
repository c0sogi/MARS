import os
import pandas as pd
import numpy as np
from library.config import INPUT_DIR, CACHE_DIR, GNSS_COLS, IMU_COLS, WGS84_A, WGS84_B
from library.utils import ecef_to_lla


def aggregate_gnss(gnss_df):
    """
    Aggregates GNSS data by timestamp.
    Computes mean, std, max, min for signal strength (Cn0DbHz).
    Computes mean, std for elevation.
    Counts number of satellites.
    """
    # Group by timestamp
    grouped = gnss_df.groupby("utcTimeMillis")

    # Aggregations
    agg_df = grouped.agg(
        {
            "Cn0DbHz": ["mean", "std", "max", "min"],
            "SvElevationDegrees": ["mean", "std"],
            "Svid": "count",
        }
    )

    # Flatten columns
    agg_df.columns = [
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_max",
        "Cn0DbHz_min",
        "SvElevationDegrees_mean",
        "SvElevationDegrees_std",
        "sv_count",
    ]

    return agg_df


def aggregate_imu(imu_df):
    """
    Aggregates IMU data (Accelerometer) by timestamp.
    Computes magnitude of acceleration and its statistics.
    """
    # Filter for Accelerometer
    acc_df = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()

    if acc_df.empty:
        return pd.DataFrame()

    # Calculate magnitude
    acc_df["magnitude"] = np.sqrt(
        acc_df["MeasurementX"] ** 2
        + acc_df["MeasurementY"] ** 2
        + acc_df["MeasurementZ"] ** 2
    )

    # Group by timestamp
    grouped = acc_df.groupby("utcTimeMillis")

    # Aggregations
    agg_df = grouped.agg({"magnitude": ["mean", "std"]})

    # Flatten columns
    agg_df.columns = ["imu_acc_mag_mean", "imu_acc_mag_std"]

    return agg_df


def process_trip(trip_id, trip_meta_df):
    """
    Processes a single trip: loads raw data, aggregates, merges with metadata.
    """
    # Extract path info from the first row of metadata for this trip
    first_row = trip_meta_df.iloc[0]
    gnss_rel_path = first_row["gnss_path"]
    imu_rel_path = first_row["imu_path"]

    gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)
    imu_path = os.path.join(INPUT_DIR, imu_rel_path)

    # --- Load GNSS ---
    # We need WLS positions and signal features
    try:
        gnss_df = pd.read_csv(gnss_path, usecols=GNSS_COLS)
    except Exception as e:
        print(f"Error loading GNSS for {trip_id}: {e}")
        return pd.DataFrame()

    # --- Load IMU ---
    try:
        imu_df = pd.read_csv(imu_path, usecols=IMU_COLS)
    except Exception as e:
        print(f"Error loading IMU for {trip_id}: {e}")
        # Continue without IMU if missing
        imu_df = pd.DataFrame(columns=IMU_COLS)

    # --- Aggregate Features ---
    gnss_feats = aggregate_gnss(gnss_df)
    imu_feats = aggregate_imu(imu_df)

    # --- Extract Baseline WLS Position ---
    # WLS position is repeated for every satellite at the same timestamp.
    # We drop duplicates to get one position per timestamp.
    wls_df = (
        gnss_df[
            [
                "utcTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .drop_duplicates(subset="utcTimeMillis")
        .set_index("utcTimeMillis")
    )

    # Convert ECEF to LLA
    lat, lon, alt = ecef_to_lla(
        wls_df["WlsPositionXEcefMeters"].values,
        wls_df["WlsPositionYEcefMeters"].values,
        wls_df["WlsPositionZEcefMeters"].values,
    )

    wls_df["lat_wls"] = lat
    wls_df["lon_wls"] = lon
    wls_df["alt_wls"] = alt

    # Drop ECEF columns to save space
    wls_df = wls_df.drop(
        columns=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    )

    # --- Merge Everything ---
    # Join features to WLS positions (all indexed by utcTimeMillis)
    trip_features = wls_df.join(gnss_feats, how="left").join(imu_feats, how="left")

    # Reset index to make 'utcTimeMillis' a column for merging with metadata
    trip_features = trip_features.reset_index().rename(
        columns={"utcTimeMillis": "UnixTimeMillis"}
    )

    # --- Merge with Metadata (Ground Truth / Targets) ---
    # The metadata contains the specific timestamps we need to predict/train on.
    # We perform an inner join to keep only the relevant rows.
    merged_df = pd.merge(trip_meta_df, trip_features, on="UnixTimeMillis", how="inner")

    # Drop rows where baseline WLS position is missing to prevent NaN targets/metrics
    # Cite debug_lesson_2: Sanitize Regression Targets, Not Just Input Features
    merged_df = merged_df.dropna(subset=["lat_wls", "lon_wls"])

    # --- Calculate Targets (if GT is present) ---
    if (
        "LatitudeDegrees" in merged_df.columns
        and "LongitudeDegrees" in merged_df.columns
    ):
        merged_df["lat_error"] = merged_df["LatitudeDegrees"] - merged_df["lat_wls"]
        merged_df["lon_error"] = merged_df["LongitudeDegrees"] - merged_df["lon_wls"]

    return merged_df


def generate_dataset(metadata_path, load_cached_data=True, split_name="train"):
    """
    Generates the dataset for a specific split (train, val, test).
    Uses caching to avoid re-processing.
    """
    cache_file = os.path.join(CACHE_DIR, f"{split_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split_name} data from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating {split_name} data from scratch...")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Process each trip
    trip_ids = meta_df["tripId"].unique()
    processed_dfs = []

    for trip_id in trip_ids:
        trip_meta = meta_df[meta_df["tripId"] == trip_id].copy()
        trip_data = process_trip(trip_id, trip_meta)
        if not trip_data.empty:
            processed_dfs.append(trip_data)

    if not processed_dfs:
        print("Warning: No data processed.")
        return pd.DataFrame()

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Save to cache
    print(f"Saving {split_name} data to {cache_file}...")
    full_df.to_parquet(cache_file, index=False)

    return full_df
