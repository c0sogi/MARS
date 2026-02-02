import os
import pandas as pd
import numpy as np
from library import config
from library.utils import ecef_to_geodetic


def aggregate_gnss_imu(gnss_df, imu_df):
    """
    Aggregates raw GNSS and IMU data by timestamp to create feature vectors.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS measurements.
        imu_df (pd.DataFrame): Raw IMU measurements.

    Returns:
        pd.DataFrame: Aggregated features indexed by utcTimeMillis.
    """
    # --- GNSS Aggregation ---
    # Group by timestamp
    gnss_grouped = gnss_df.groupby("utcTimeMillis")

    # Compute statistics for signal strength and satellite count
    gnss_feats = gnss_grouped.agg(
        {
            "Cn0DbHz": ["mean", "std", "max"],
            "Svid": "count",
            "SvElevationDegrees": "mean",
        }
    )

    # Flatten MultiIndex columns
    gnss_feats.columns = [
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_max",
        "Svid_count",
        "SvElevationDegrees_mean",
    ]

    # Handle NaNs created by std of single element groups
    gnss_feats["Cn0DbHz_std"] = gnss_feats["Cn0DbHz_std"].fillna(0)

    # --- IMU Aggregation ---
    # Filter for Uncalibrated Accelerometer data
    if imu_df is not None and not imu_df.empty:
        accel_df = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
        if not accel_df.empty:
            # Calculate magnitude of acceleration vector
            accel_df["Accel_mag"] = np.sqrt(
                accel_df["MeasurementX"] ** 2
                + accel_df["MeasurementY"] ** 2
                + accel_df["MeasurementZ"] ** 2
            )

            imu_grouped = accel_df.groupby("utcTimeMillis")
            imu_feats = imu_grouped.agg({"Accel_mag": ["mean", "std"]})
            imu_feats.columns = ["Accel_mag_mean", "Accel_mag_std"]
            imu_feats["Accel_mag_std"] = imu_feats["Accel_mag_std"].fillna(0)
        else:
            imu_feats = pd.DataFrame()
    else:
        imu_feats = pd.DataFrame()

    # --- Merge ---
    # GNSS is the primary time source. Left join IMU data.
    features = gnss_feats.join(imu_feats, how="left")

    # Ensure all expected features exist (Cite debug_lesson_3: Robustness against missing features)
    for col in config.FEATURE_NAMES:
        if col not in features.columns:
            features[col] = 0.0

    # Fill missing IMU data with 0 (assuming no motion/variance if missing)
    features.fillna(0, inplace=True)

    return features


def prepare_trip_data(meta_trip_df):
    """
    Loads raw data for a single trip, computes aggregated features,
    aligns with WLS baseline, and calculates residual targets.

    Args:
        meta_trip_df (pd.DataFrame): Metadata rows for a single trip.

    Returns:
        pd.DataFrame: Processed dataframe for the trip.
    """
    # Extract file paths from the first row of metadata (constant for the trip)
    first_row = meta_trip_df.iloc[0]
    gnss_rel_path = first_row["gnss_path"]
    imu_rel_path = first_row["imu_path"]

    gnss_path = os.path.join(config.INPUT_DIR, gnss_rel_path)
    imu_path = os.path.join(config.INPUT_DIR, imu_rel_path)

    # Load GNSS Data
    # We need raw measurement columns for features and WLS columns for baseline
    gnss_cols_needed = config.GNSS_RAW_COLS + [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Robust loading: read only needed columns if possible, else read all
    try:
        gnss_df = pd.read_csv(gnss_path, usecols=lambda c: c in set(gnss_cols_needed))
    except ValueError:
        gnss_df = pd.read_csv(gnss_path)

    # Load IMU Data
    try:
        imu_df = pd.read_csv(imu_path, usecols=lambda c: c in set(config.IMU_RAW_COLS))
    except ValueError:
        imu_df = pd.read_csv(imu_path)

    # 1. Aggregate Features (Raw GNSS + IMU)
    features_df = aggregate_gnss_imu(gnss_df, imu_df)

    # 2. Get WLS Baseline Position
    # WLS position is repeated for every satellite measurement in an epoch.
    # We take the first valid position for each timestamp.
    wls_df = gnss_df.groupby("utcTimeMillis")[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].first()

    # Convert WLS ECEF coordinates to Geodetic (Lat/Lon)
    wls_lat, wls_lon, _ = ecef_to_geodetic(
        wls_df["WlsPositionXEcefMeters"].values,
        wls_df["WlsPositionYEcefMeters"].values,
        wls_df["WlsPositionZEcefMeters"].values,
    )

    wls_df["lat_wls"] = wls_lat
    wls_df["lon_wls"] = wls_lon

    # 3. Merge Features with Baseline
    # Index is utcTimeMillis for both
    trip_data = features_df.join(wls_df[["lat_wls", "lon_wls"]], how="inner")

    # 4. Merge with Metadata (Ground Truth / Target Timestamps)
    # Metadata contains the ground truth Lat/Lon and the specific timestamps we need.
    # Rename UnixTimeMillis to utcTimeMillis for joining
    meta_trip_df = meta_trip_df.rename(columns={"UnixTimeMillis": "utcTimeMillis"})
    meta_trip_df = meta_trip_df.set_index("utcTimeMillis")

    # Inner join filters the raw data to only the timestamps present in metadata
    final_df = trip_data.join(meta_trip_df, how="inner", rsuffix="_meta")

    # 5. Calculate Residual Targets
    # Target = Ground Truth - WLS Baseline
    if "LatitudeDegrees" in final_df.columns:
        final_df["dLat"] = final_df["LatitudeDegrees"] - final_df["lat_wls"]
        final_df["dLon"] = final_df["LongitudeDegrees"] - final_df["lon_wls"]
    else:
        # For test set, targets are 0 (placeholders)
        final_df["dLat"] = 0.0
        final_df["dLon"] = 0.0

    # Reset index to make utcTimeMillis a column
    final_df = final_df.reset_index()

    # Ensure tripId is populated (it might be in index or column depending on join)
    if "tripId" not in final_df.columns and "tripId" in meta_trip_df.columns:
        final_df["tripId"] = meta_trip_df["tripId"].iloc[0]

    # Keep only necessary columns
    cols_to_keep = config.FEATURE_NAMES + [
        "lat_wls",
        "lon_wls",
        "dLat",
        "dLon",
        "tripId",
        "utcTimeMillis",
    ]
    # Filter to available columns
    available_cols = [c for c in cols_to_keep if c in final_df.columns]

    return final_df[available_cols]


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Main function to process the dataset.
    Loads metadata, processes each trip, concatenates results, and handles caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to save/load the parquet cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The complete processed dataset.
    """
    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing dataset from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    # Group metadata by tripId to process one trip at a time
    trips = meta_df.groupby("tripId")

    all_trips_data = []

    # Iterate over all trips
    for trip_id, trip_meta in trips:
        try:
            trip_data = prepare_trip_data(trip_meta)
            all_trips_data.append(trip_data)
        except Exception as e:
            print(f"Error processing trip {trip_id}: {e}")
            continue

    if not all_trips_data:
        raise ValueError("No data processed! Check input paths and metadata.")

    # Concatenate all trips into a single DataFrame
    full_df = pd.concat(all_trips_data, ignore_index=True)

    # Sort by tripId and timestamp to ensure sequential order
    full_df = full_df.sort_values(["tripId", "utcTimeMillis"]).reset_index(drop=True)

    # Save to cache
    print(f"Saving processed data to {cache_path}")
    full_df.to_parquet(cache_path)

    return full_df
