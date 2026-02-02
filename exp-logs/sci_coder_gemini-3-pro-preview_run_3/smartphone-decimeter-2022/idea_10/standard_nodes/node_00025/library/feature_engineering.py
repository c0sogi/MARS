import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import WORKING_DIR, INPUT_DIR, SEED
from library.utils import ecef_to_enu, geodetic_to_ecef
from library.data_loader import (
    load_metadata,
    load_gnss_raw,
    load_imu_raw,
    load_ground_truth,
)


def estimate_doppler_velocity(gnss_df):
    """
    Estimates the receiver velocity using the Doppler shift (Pseudorange Rate)
    via Weighted Least Squares (WLS).

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data containing satellite states and measurements.

    Returns:
        pd.DataFrame: DataFrame with estimated velocity vectors [vx, vy, vz] and uncertainty.
    """
    # Filter valid measurements for Doppler computation
    # Requirements: Valid PseudorangeRate, Valid Satellite Velocity, Valid WLS Position
    required_cols = [
        "PseudorangeRateMetersPerSecond",
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Drop rows where any required column is NaN
    valid_mask = gnss_df[required_cols].notna().all(axis=1)
    df = gnss_df[valid_mask].copy()

    # Calculate Line-of-Sight (LOS) vectors from Receiver to Satellite
    # P_sat - P_rx
    dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
    dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
    dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Unit vectors
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    # Calculate Satellite Velocity projected onto LOS
    # v_sat dot u
    v_sat_los = (
        df["SvVelocityXEcefMetersPerSecond"] * ux
        + df["SvVelocityYEcefMetersPerSecond"] * uy
        + df["SvVelocityZEcefMetersPerSecond"] * uz
    )

    # Measurement y = v_sat_los - PseudorangeRate
    # Equation: PseudorangeRate = (v_sat - v_rx) dot u + clock_drift
    # Rearranged: v_rx dot u - clock_drift = v_sat dot u - PseudorangeRate
    df["y_obs"] = v_sat_los - df["PseudorangeRateMetersPerSecond"]

    # Weights: Inverse variance
    # Handle zero uncertainty to avoid div by zero
    unc = df["PseudorangeRateUncertaintyMetersPerSecond"].fillna(1.0)
    unc = np.where(unc <= 0, 1.0, unc)
    df["weight"] = 1.0 / (unc**2)

    # Assign unit vectors to dataframe for grouping
    df["ux"] = ux
    df["uy"] = uy
    df["uz"] = uz

    # Group by epoch to solve WLS
    # We process each (drive, phone, time) group
    # Using pandas groupby apply is slow for WLS, so we iterate or use optimized approach.
    # Given the constraints, we'll use a custom aggregation or iteration.
    # Iteration over groups is safer for correctness.

    results = []

    # Group keys
    keys = ["drive_id", "phone_name", "UnixTimeMillis"]
    grouped = df.groupby(keys)

    # Iterate through groups (this can be slow, but robust)
    # To optimize, we could vectorize, but variable number of sats makes it tricky.
    # We will use a simplified loop.

    for name, group in tqdm(grouped, desc="Estimating Doppler Velocity", leave=False):
        if len(group) < 4:
            # Not enough satellites for 4 unknowns (vx, vy, vz, drift)
            results.append(list(name) + [np.nan, np.nan, np.nan])
            continue

        # H matrix: [ux, uy, uz, -1]
        H = group[["ux", "uy", "uz"]].values
        H = np.hstack([H, -np.ones((len(H), 1))])

        y = group["y_obs"].values
        W = np.diag(group["weight"].values)

        # Solve: (H^T W H) x = H^T W y
        try:
            HTW = H.T @ W
            HTWH = HTW @ H
            HTWy = HTW @ y

            # Add regularization for stability
            HTWH_reg = HTWH + np.eye(4) * 1e-3

            x_sol = np.linalg.solve(HTWH_reg, HTWy)

            # x_sol = [vx, vy, vz, drift]
            results.append(list(name) + [x_sol[0], x_sol[1], x_sol[2]])
        except np.linalg.LinAlgError:
            results.append(list(name) + [np.nan, np.nan, np.nan])

    result_df = pd.DataFrame(
        results, columns=keys + ["v_doppler_x", "v_doppler_y", "v_doppler_z"]
    )
    return result_df


def create_pointwise_features(gnss_df, imu_df):
    """
    Creates aggregated features for each timestamp.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data.
        imu_df (pd.DataFrame): Raw IMU data.

    Returns:
        pd.DataFrame: Feature dataframe.
    """
    # --- GNSS Aggregation ---
    # Group by epoch
    keys = ["drive_id", "phone_name", "UnixTimeMillis"]

    # Define aggregations
    gnss_aggs = {
        "Cn0DbHz": ["mean", "max", "std"],
        "Svid": ["count"],
        "PseudorangeRateUncertaintyMetersPerSecond": ["mean"],
        "SvElevationDegrees": ["mean"],
    }

    gnss_feats = gnss_df.groupby(keys).agg(gnss_aggs)
    gnss_feats.columns = [f"gnss_{c[0]}_{c[1]}" for c in gnss_feats.columns]
    gnss_feats.reset_index(inplace=True)

    # --- IMU Aggregation ---
    # IMU is high frequency. We align to GNSS seconds.
    # Create a merge key: round UnixTimeMillis to nearest second?
    # The GNSS data is typically 1Hz.
    # Let's assume we aggregate IMU over the 1-second window matching the GNSS timestamp.
    # We create a 'merge_time' for IMU by rounding to nearest 1000ms.

    if not imu_df.empty:
        imu_df = imu_df.copy()
        # IMU utcTimeMillis is millis.
        imu_df["merge_time"] = (
            np.round(imu_df["UnixTimeMillis"] / 1000) * 1000
        ).astype(np.int64)

        # Calculate magnitude
        imu_df["magnitude"] = np.sqrt(
            imu_df["MeasurementX"] ** 2
            + imu_df["MeasurementY"] ** 2
            + imu_df["MeasurementZ"] ** 2
        )

        # Pivot or filter by MessageType
        # We mainly care about Accelerometer (UncalAccel) for motion state
        accel_df = imu_df[imu_df["MessageType"] == "UncalAccel"]

        imu_aggs = {"magnitude": ["mean", "std", "max"]}

        # Group by drive, phone, and the aligned time
        imu_feats = accel_df.groupby(["drive_id", "phone_name", "merge_time"]).agg(
            imu_aggs
        )
        imu_feats.columns = [f"imu_acc_{c[0]}_{c[1]}" for c in imu_feats.columns]
        imu_feats.reset_index(inplace=True)
        imu_feats.rename(columns={"merge_time": "UnixTimeMillis"}, inplace=True)

        # Merge GNSS and IMU
        features = pd.merge(gnss_feats, imu_feats, on=keys, how="left")
    else:
        features = gnss_feats

    return features


def prepare_dataset(split_name, load_cached_data=True):
    """
    Prepares the dataset for training or inference.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (features_df, targets_df) for train/val, or (features_df, None) for test.
    """
    cache_file = os.path.join(WORKING_DIR, f"{split_name}_dataset.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading prepared {split_name} dataset from cache...")
        df = pd.read_parquet(cache_file)

        # Split features and targets if applicable
        target_cols = ["target_e", "target_n", "target_u"]
        if all(c in df.columns for c in target_cols):
            return df.drop(columns=target_cols), df[target_cols]
        else:
            return df, None

    print(f"Preparing {split_name} dataset from scratch...")

    # 1. Load Metadata
    meta_df = load_metadata(split_name)

    # 2. Load Raw Data
    gnss_df = load_gnss_raw(meta_df, split_name, load_cached_data=load_cached_data)
    imu_df = load_imu_raw(meta_df, split_name, load_cached_data=load_cached_data)

    # 3. Compute Doppler Velocity
    print("Computing Doppler Velocity...")
    doppler_df = estimate_doppler_velocity(gnss_df)

    # 4. Compute Point-wise Features
    print("Computing Point-wise Features...")
    features_df = create_pointwise_features(gnss_df, imu_df)

    # 5. Merge Doppler and Features
    full_df = pd.merge(
        features_df,
        doppler_df,
        on=["drive_id", "phone_name", "UnixTimeMillis"],
        how="left",
    )

    # 6. Add WLS Baseline Position (needed for global optimization later and target calc)
    # We take the first WLS position for each timestamp (they are repeated per satellite)
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    wls_df = (
        gnss_df.groupby(["drive_id", "phone_name", "UnixTimeMillis"])[wls_cols]
        .first()
        .reset_index()
    )

    full_df = pd.merge(
        full_df, wls_df, on=["drive_id", "phone_name", "UnixTimeMillis"], how="left"
    )

    # 7. Handle Targets (Train/Val only)
    if split_name in ["train", "val"]:
        print("Loading Ground Truth and Computing Targets...")
        gt_df = load_ground_truth(
            meta_df, split_name, load_cached_data=load_cached_data
        )

        # Merge GT with features
        # Note: GT timestamps might not perfectly align with GNSS if there are gaps,
        # but metadata ensures we only look at timestamps present in GT.
        # However, our features are built from GNSS. We should inner join with metadata timestamps.

        # Filter features to only those in metadata (which are the target timestamps)
        # The metadata df has the exact timestamps we need.
        # We merge with gt_df to get AltitudeMeters which is not in meta_df
        target_keys = meta_df[["drive_id", "phone_name", "UnixTimeMillis"]]

        target_timestamps = pd.merge(
            target_keys,
            gt_df[
                [
                    "drive_id",
                    "phone_name",
                    "UnixTimeMillis",
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "AltitudeMeters",
                ]
            ],
            on=["drive_id", "phone_name", "UnixTimeMillis"],
            how="left",
        )

        # Merge features onto target timestamps
        dataset = pd.merge(
            target_timestamps,
            full_df,
            on=["drive_id", "phone_name", "UnixTimeMillis"],
            how="left",
        )

        # Calculate ENU Residuals

        # WLS positions might be NaN if GNSS failed. Drop these rows.
        # We must drop them before computing targets because we need WLS to compute residuals.
        dataset = dataset.dropna(subset=wls_cols)

        # 1. Convert GT Lat/Lon/Alt to ECEF
        # Fill missing Altitude with 0 or mean (it's noisy anyway, but needed for conversion)
        dataset["AltitudeMeters"] = dataset["AltitudeMeters"].fillna(0)

        gt_x, gt_y, gt_z = geodetic_to_ecef(
            dataset["LatitudeDegrees"].values,
            dataset["LongitudeDegrees"].values,
            dataset["AltitudeMeters"].values,
        )

        # 2. Calculate ECEF Residuals
        res_x = gt_x - dataset["WlsPositionXEcefMeters"].values
        res_y = gt_y - dataset["WlsPositionYEcefMeters"].values
        res_z = gt_z - dataset["WlsPositionZEcefMeters"].values

        # 3. Rotate to ENU (using WLS position as reference origin)
        # We need WLS Lat/Lon for the rotation matrix
        # Since we don't have WLS Lat/Lon directly, we can use the GT Lat/Lon as the reference frame origin
        # for the rotation matrix. The error is small enough (local tangent plane).
        # Or convert WLS ECEF to Geodetic. Let's use GT as ref for rotation frame stability.

        e, n, u = ecef_to_enu(
            gt_x,
            gt_y,
            gt_z,
            dataset["LatitudeDegrees"].values,
            dataset["LongitudeDegrees"].values,
            dataset["AltitudeMeters"].values,
        )

        # Wait, ecef_to_enu(x, y, z, ref_lat, ref_lon) converts point (x,y,z) to ENU relative to ref.
        # We want the vector (res_x, res_y, res_z) in ENU.
        # This is equivalent to converting (WLS_x, WLS_y, WLS_z) to ENU relative to GT,
        # then taking the negative? Or converting GT to ENU relative to WLS?
        # Let's define Target = GT - WLS (in ENU frame).
        # So we convert GT position to ENU relative to WLS position?
        # No, usually we define a local tangent plane at the approximate position (WLS).
        # Let's convert WLS ECEF to WLS Lat/Lon first to establish the local frame.
        # Since we don't have that function exposed easily and don't want to reimplement complex logic,
        # Using GT as the reference frame origin is acceptable for small residuals.
        # Let's compute the vector in ECEF: Vec = GT - WLS.
        # Rotate this vector into the ENU frame defined at GT.

        # Rotation matrix from ECEF to ENU at (lat, lon):
        # R = [[-sin_lon,           cos_lon,          0],
        #      [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        #      [ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat]]
        # enu = R * ecef_vec

        lat_rad = np.radians(dataset["LatitudeDegrees"].values)
        lon_rad = np.radians(dataset["LongitudeDegrees"].values)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Vector from WLS to GT (Target correction)
        # Target = GT - WLS
        dx = res_x  # GT - WLS
        dy = res_y
        dz = res_z

        target_e = -sin_lon * dx + cos_lon * dy
        target_n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        target_u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        dataset["target_e"] = target_e
        dataset["target_n"] = target_n
        dataset["target_u"] = target_u

        # Save cache
        print(f"Saving {split_name} dataset to cache...")
        dataset.to_parquet(cache_file, index=False)

        return (
            dataset.drop(columns=["target_e", "target_n", "target_u"]),
            dataset[["target_e", "target_n", "target_u"]],
        )

    else:
        # Test split
        # We need to ensure we have rows for all timestamps in metadata
        target_timestamps = meta_df[
            ["tripId", "drive_id", "phone_name", "UnixTimeMillis"]
        ]
        dataset = pd.merge(
            target_timestamps,
            full_df,
            on=["drive_id", "phone_name", "UnixTimeMillis"],
            how="left",
        )

        # Fill missing WLS with forward fill then backward fill (if any)
        # WLS is critical for final reconstruction
        dataset[wls_cols] = dataset.groupby("tripId")[wls_cols].ffill().bfill()

        print(f"Saving {split_name} dataset to cache...")
        dataset.to_parquet(cache_file, index=False)

        return dataset, None
