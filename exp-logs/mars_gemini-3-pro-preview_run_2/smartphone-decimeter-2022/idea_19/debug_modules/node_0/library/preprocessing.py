import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import provided modules
from library.config import config
from library.utils import ecef_to_lla, degrees_to_meters

# Suppress warnings
warnings.filterwarnings("ignore")


def compute_dynamics(df):
    """
    Compute velocity features (first order differences) in meters.
    """
    # Ensure WLS LLA exists
    if "wls_lat" not in df.columns:
        lat, lon, alt = ecef_to_lla(
            df["WlsPositionXEcefMeters"].values,
            df["WlsPositionYEcefMeters"].values,
            df["WlsPositionZEcefMeters"].values,
        )
        df["wls_lat"] = lat
        df["wls_lon"] = lon
        df["wls_alt"] = alt

    # Calculate time difference in seconds
    dt = df["utcTimeMillis"].diff() / 1000.0
    dt = dt.fillna(1.0)  # Avoid division by zero or NaN at start
    # Replace 0 with 1.0 to avoid inf (duplicate timestamps shouldn't exist but safety first)
    dt = dt.replace(0, 1.0)

    # Calculate differences
    d_lat = df["wls_lat"].diff().fillna(0)
    d_lon = df["wls_lon"].diff().fillna(0)
    d_alt = df["wls_alt"].diff().fillna(0)

    # Convert degrees to meters
    # Use current latitude for longitude scaling
    north_m, east_m = degrees_to_meters(
        d_lat.values, d_lon.values, df["wls_lat"].values
    )

    # Velocity
    df["vel_lat_m"] = north_m / dt.values
    df["vel_lon_m"] = east_m / dt.values
    df["vel_alt_m"] = d_alt / dt.values

    # Relative positions (will be handled by windowing in Dataset,
    # but we need the base WLS LLA columns for that)
    # We return the dataframe with velocity columns added

    return df


def compute_satellite_stats(gnss_df):
    """
    Aggregate satellite geometry statistics per epoch.
    """
    # Group by epoch
    grp = gnss_df.groupby("utcTimeMillis")

    stats = grp.agg(
        {
            "SvElevationDegrees": ["mean", "std"],
            "SvAzimuthDegrees": ["mean", "std"],
            "Cn0DbHz": "mean",
            "RawPseudorangeUncertaintyMeters": "mean",
        }
    )

    stats.columns = ["mean_elev", "std_elev", "mean_azim", "std_azim", "cn0", "unc_m"]

    # Fill NaNs (e.g. std of single satellite)
    stats = stats.fillna(0)

    return stats.reset_index()


def resample_imu(imu_df, gnss_timestamps):
    """
    Resample IMU data to align with GNSS timestamps.
    """
    if imu_df.empty:
        # Return empty dataframe with expected columns
        cols = [
            "utcTimeMillis",
            "mean_acc_mag",
            "std_acc_mag",
            "mean_gyro_mag",
            "std_gyro_mag",
        ]
        return pd.DataFrame(columns=cols)

    # Calculate magnitudes
    # Accelerometer
    acc_mask = imu_df["MessageType"] == "UncalAccel"
    acc_df = imu_df[acc_mask].copy()
    acc_df["mag"] = np.sqrt(
        acc_df["MeasurementX"] ** 2
        + acc_df["MeasurementY"] ** 2
        + acc_df["MeasurementZ"] ** 2
    )

    # Gyroscope
    gyro_mask = imu_df["MessageType"] == "UncalGyro"
    gyro_df = imu_df[gyro_mask].copy()
    gyro_df["mag"] = np.sqrt(
        gyro_df["MeasurementX"] ** 2
        + gyro_df["MeasurementY"] ** 2
        + gyro_df["MeasurementZ"] ** 2
    )

    # Prepare GNSS timestamps dataframe
    gnss_ts_df = pd.DataFrame({"utcTimeMillis": gnss_timestamps})
    gnss_ts_df = gnss_ts_df.sort_values("utcTimeMillis").reset_index(drop=True)

    # Function to aggregate using merge_asof
    def aggregate_imu(source_df, name_prefix):
        if source_df.empty:
            return pd.DataFrame(
                {
                    "utcTimeMillis": gnss_timestamps,
                    f"mean_{name_prefix}": 0.0,
                    f"std_{name_prefix}": 0.0,
                }
            )

        source_df = source_df.sort_values("utcTimeMillis")

        # We assign each IMU measurement to the nearest GNSS epoch within 500ms
        imu_labeled = pd.merge_asof(
            source_df[["utcTimeMillis", "mag"]],
            gnss_ts_df.rename(columns={"utcTimeMillis": "gnss_time"}),
            left_on="utcTimeMillis",
            right_on="gnss_time",
            direction="nearest",
            tolerance=500,
        )

        # Drop unmatched IMU rows
        imu_labeled = imu_labeled.dropna(subset=["gnss_time"])

        # Aggregate
        agg = imu_labeled.groupby("gnss_time")["mag"].agg(["mean", "std"]).reset_index()
        agg.columns = ["utcTimeMillis", f"mean_{name_prefix}", f"std_{name_prefix}"]

        # Merge back to original GNSS timestamps to ensure all are present
        result = pd.merge(gnss_ts_df, agg, on="utcTimeMillis", how="left").fillna(0)
        return result

    acc_agg = aggregate_imu(acc_df, "acc_mag")
    gyro_agg = aggregate_imu(gyro_df, "gyro_mag")

    # Combine
    imu_features = pd.merge(acc_agg, gyro_agg, on="utcTimeMillis", how="inner")

    return imu_features


def process_trip(trip_id, gnss_path, imu_path, gt_df=None):
    """
    Process a single trip: load, clean, feature engineer, and merge.
    """
    # Load GNSS
    try:
        gnss_df = pd.read_csv(
            os.path.join(config.INPUT_DIR, gnss_path), usecols=config.GNSS_COLS
        )
    except Exception as e:
        print(f"Error loading GNSS for {trip_id}: {e}")
        return None

    # Drop duplicates
    gnss_df = gnss_df.drop_duplicates(subset=["utcTimeMillis", "Svid"])

    # 1. Satellite Stats (Environmental Context)
    # This aggregates from signal level to epoch level
    sat_stats = compute_satellite_stats(gnss_df)

    # 2. WLS Position (Trajectory Base)
    # Get the first WLS position per epoch (they are repeated per signal)
    wls_df = (
        gnss_df.groupby("utcTimeMillis")[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .first()
        .reset_index()
    )

    # 3. Dynamics
    # Convert WLS to LLA and compute velocities
    traj_df = compute_dynamics(wls_df)

    # 4. IMU (Inertial Context)
    if os.path.exists(os.path.join(config.INPUT_DIR, imu_path)):
        try:
            imu_df = pd.read_csv(
                os.path.join(config.INPUT_DIR, imu_path), usecols=config.IMU_COLS
            )
            imu_feats = resample_imu(imu_df, traj_df["utcTimeMillis"].values)
        except Exception as e:
            print(f"Error loading IMU for {trip_id}: {e}")
            # Create zero features
            imu_feats = pd.DataFrame({"utcTimeMillis": traj_df["utcTimeMillis"]})
            for col in config.IMU_FEATURES:
                imu_feats[col] = 0.0
    else:
        imu_feats = pd.DataFrame({"utcTimeMillis": traj_df["utcTimeMillis"]})
        for col in config.IMU_FEATURES:
            imu_feats[col] = 0.0

    # Merge all features
    # Base is traj_df (epochs)
    df = pd.merge(traj_df, sat_stats, on="utcTimeMillis", how="left")
    df = pd.merge(df, imu_feats, on="utcTimeMillis", how="left")

    # Add Metadata
    df["tripId"] = trip_id

    # 5. Targets (if GT provided)
    if gt_df is not None:
        # Merge GT
        # GT has UnixTimeMillis
        df = pd.merge(
            df,
            gt_df[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            left_on="utcTimeMillis",
            right_on="UnixTimeMillis",
            how="inner",
        )

        # Calculate residuals in meters
        # Target = GT - WLS
        d_lat = df["LatitudeDegrees"] - df["wls_lat"]
        d_lon = df["LongitudeDegrees"] - df["wls_lon"]

        north_m, east_m = degrees_to_meters(
            d_lat.values, d_lon.values, df["wls_lat"].values
        )

        df["target_north_m"] = north_m
        df["target_east_m"] = east_m

        # Drop GT columns to save space, keep only targets
        df = df.drop(columns=["LatitudeDegrees", "LongitudeDegrees", "UnixTimeMillis"])

    return df


def preprocess_dataset(metadata_path, mode="train", load_cached_data=True):
    """
    Main preprocessing function.

    Args:
        metadata_path: Path to metadata CSV.
        mode: 'train', 'val', or 'test'.
        load_cached_data: Whether to try loading from cache.

    Returns:
        If mode in ['train', 'val']:
            X (pd.DataFrame): Feature dataframe.
            y (np.array): Target array (N, 2).
        If mode == 'test':
            X (pd.DataFrame): Feature dataframe (includes metadata cols).
    """
    cache_dir = config.WORKING_DIR
    cache_file = os.path.join(cache_dir, f"{mode}_data.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            if mode in ["train", "val"]:
                target_cols = ["target_north_m", "target_east_m"]
                y = df[target_cols].values
                X = df.drop(columns=target_cols)
                return X, y
            else:
                return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {mode} data from scratch...")
    meta_df = pd.read_csv(metadata_path)

    # For debugging, sample a subset
    if config.DEBUG:
        print("DEBUG MODE: Sampling 5 trips.")
        trips = meta_df["tripId"].unique()[:5]
        meta_df = meta_df[meta_df["tripId"].isin(trips)]

    unique_trips = meta_df["tripId"].unique()
    results = []

    for trip_id in tqdm(unique_trips, desc=f"Processing {mode} trips"):
        trip_meta = meta_df[meta_df["tripId"] == trip_id]

        # Get file paths (take first row)
        row = trip_meta.iloc[0]
        gnss_path = row["gnss_path"]
        imu_path = row["imu_path"]

        # For train/val, we pass the GT dataframe to process_trip to compute targets
        gt_df = trip_meta if mode in ["train", "val"] else None

        processed_df = process_trip(trip_id, gnss_path, imu_path, gt_df)

        if processed_df is not None and not processed_df.empty:
            results.append(processed_df)

    if not results:
        raise ValueError("No data processed!")

    full_df = pd.concat(results, ignore_index=True)

    # Fill any remaining NaNs (e.g. from dynamics diffs at start of trip)
    full_df = full_df.fillna(0)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_file}")
    full_df.to_parquet(cache_file, index=False)

    if mode in ["train", "val"]:
        target_cols = ["target_north_m", "target_east_m"]
        y = full_df[target_cols].values
        X = full_df.drop(columns=target_cols)
        return X, y
    else:
        return full_df
