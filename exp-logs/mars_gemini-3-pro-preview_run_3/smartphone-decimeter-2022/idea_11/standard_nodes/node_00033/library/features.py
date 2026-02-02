import os
import numpy as np
import pandas as pd
import math
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    FEATURE_COLS,
    TARGET_COLS,
    WGS84_A,
    WGS84_F,
    WGS84_B,
)
from library.utils import (
    wgs84_to_ecef,
    ecef_to_enu,
    euclidean_distance,
    calculate_los_vector,
    project_velocity,
)


def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Latitude, Longitude, Altitude.
    Vectorized implementation.
    """
    # WGS84 constants
    a = WGS84_A
    b = WGS84_B
    e = np.sqrt(1 - (b / a) ** 2)
    ep = np.sqrt((a / b) ** 2 - 1)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep**2 * b * np.sin(th) ** 3, p - e**2 * a * np.cos(th) ** 3)

    # Iterative refinement not strictly necessary for meter-level accuracy WLS,
    # but standard formula is usually sufficient.

    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def compute_geometric_residuals(df):
    """
    Compute Geometric Consistency (Post-Fit Residuals).
    Residual = RawPseudorange - Distance(User, Sat) - ClockBias
    """
    # Calculate Euclidean distance from WLS position to Satellite
    dist = euclidean_distance(
        df["WlsPositionXEcefMeters"],
        df["WlsPositionYEcefMeters"],
        df["WlsPositionZEcefMeters"],
        df["SvPositionXEcefMeters"],
        df["SvPositionYEcefMeters"],
        df["SvPositionZEcefMeters"],
    )

    # Raw residual
    raw_residual = df["RawPseudorangeMeters"] - dist

    # Estimate Clock Bias per epoch (median of residuals)
    # We use transform to broadcast the median back to original rows
    clock_bias = df.groupby("UnixTimeMillis")["raw_residual"].transform("median")

    # Corrected residual
    df["pr_residual"] = raw_residual - clock_bias
    return df


def compute_doppler_residuals(df):
    """
    Compute Dynamic Consistency (Doppler Residuals).
    Residual = PseudorangeRate - ProjectedSatelliteVelocity
    (Assuming user velocity is small or captured in bias/noise for this feature)
    """
    # Calculate Line-of-Sight vector
    ux, uy, uz = calculate_los_vector(
        df["SvPositionXEcefMeters"],
        df["SvPositionYEcefMeters"],
        df["SvPositionZEcefMeters"],
        df["WlsPositionXEcefMeters"],
        df["WlsPositionYEcefMeters"],
        df["WlsPositionZEcefMeters"],
    )

    # Project Satellite Velocity onto LOS
    # Note: Pseudorange Rate is positive when distance increases (satellite moving away).
    # If sat moves away, v_sat dot u_los is positive.
    # However, standard convention and Android logs can vary.
    # We take the difference and let the model learn the magnitude of discrepancy.
    v_proj = project_velocity(
        df["SvVelocityXEcefMetersPerSecond"],
        df["SvVelocityYEcefMetersPerSecond"],
        df["SvVelocityZEcefMetersPerSecond"],
        ux,
        uy,
        uz,
    )

    # Residual
    # We take the absolute difference to capture magnitude of inconsistency
    df["doppler_residual"] = df["PseudorangeRateMetersPerSecond"] - v_proj

    return df


def process_gnss(gnss_path):
    """
    Load and process GNSS file to generate aggregated features.
    """
    try:
        # Load only necessary columns to save memory
        use_cols = [
            "utcTimeMillis",
            "Svid",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
            "RawPseudorangeMeters",
            "PseudorangeRateMetersPerSecond",
            "Cn0DbHz",
        ]
        df = pd.read_csv(gnss_path, usecols=lambda c: c in use_cols)

        # Rename for consistency
        df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

        # 1. Compute Physics Features
        df["raw_residual"] = df["RawPseudorangeMeters"] - np.sqrt(
            (df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]) ** 2
            + (df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]) ** 2
            + (df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]) ** 2
        )

        # Vectorized LOS calculation
        dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
        dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
        dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        ux, uy, uz = dx / dist, dy / dist, dz / dist

        v_proj = (
            df["SvVelocityXEcefMetersPerSecond"] * ux
            + df["SvVelocityYEcefMetersPerSecond"] * uy
            + df["SvVelocityZEcefMetersPerSecond"] * uz
        )

        df["doppler_diff"] = df["PseudorangeRateMetersPerSecond"] - v_proj

        # 2. Group by Epoch to calculate residuals and aggregates
        # We process residuals per group to remove clock bias

        # Define aggregation functions
        # For residuals, we first need to center them per group (remove clock bias)
        # Doing this inside groupby.apply is slow.
        # Faster approach: Calculate median per group using transform, then subtract.

        clock_bias = df.groupby("UnixTimeMillis")["raw_residual"].transform("median")
        df["pr_residual"] = (df["raw_residual"] - clock_bias).abs()

        # Doppler residual: The user clock drift is common to all sats.
        # We can also center this.
        clock_drift = df.groupby("UnixTimeMillis")["doppler_diff"].transform("median")
        df["doppler_residual"] = (df["doppler_diff"] - clock_drift).abs()

        # 3. Aggregate Features
        agg_funcs = {
            "Cn0DbHz": ["mean", "max", "std"],
            "Svid": "count",
            "pr_residual": ["mean", "std"],  # Using mean of abs residual as "mean_abs"
            "doppler_residual": ["mean", "std"],
        }

        df_agg = df.groupby("UnixTimeMillis").agg(agg_funcs)

        # Flatten columns
        df_agg.columns = [
            f"{c[0]}_{c[1]}" if c[1] != "count" else "sv_count" for c in df_agg.columns
        ]

        # Rename specific columns to match config
        rename_map = {
            "pr_residual_mean": "pr_residual_mean_abs",
            "doppler_residual_mean": "doppler_residual_mean_abs",
        }
        df_agg.rename(columns=rename_map, inplace=True)

        # Extract WLS position (it's constant per epoch, just take first)
        wls_pos = df.groupby("UnixTimeMillis")[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].first()

        df_agg = df_agg.join(wls_pos)

        return df_agg.reset_index()

    except Exception as e:
        print(f"Error processing GNSS {gnss_path}: {e}")
        return None


def process_imu(imu_path):
    """
    Load and process IMU file.
    Aggregates to 1-second intervals to match GNSS.
    """
    try:
        df = pd.read_csv(imu_path)

        # Calculate magnitudes
        df["mag"] = np.sqrt(
            df["MeasurementX"] ** 2 + df["MeasurementY"] ** 2 + df["MeasurementZ"] ** 2
        )

        # Pivot or filter to separate Accel and Gyro
        # MessageType: UncalAccel, UncalGyro
        accel = df[df["MessageType"] == "UncalAccel"][["utcTimeMillis", "mag"]].copy()
        gyro = df[df["MessageType"] == "UncalGyro"][["utcTimeMillis", "mag"]].copy()

        # Create merge key (seconds)
        accel["merge_key"] = (accel["utcTimeMillis"] // 1000).astype(np.int64)
        gyro["merge_key"] = (gyro["utcTimeMillis"] // 1000).astype(np.int64)

        # Aggregate
        accel_agg = accel.groupby("merge_key")["mag"].mean().rename("accel_mag_mean")
        gyro_agg = gyro.groupby("merge_key")["mag"].mean().rename("gyro_mag_mean")

        # Merge
        imu_agg = pd.concat([accel_agg, gyro_agg], axis=1).reset_index()
        return imu_agg

    except Exception as e:
        print(f"Error processing IMU {imu_path}: {e}")
        return None


def process_drive(drive_id, phone_name, gnss_rel, imu_rel, gt_rel=None, is_train=True):
    """
    Process a single drive: GNSS + IMU + (Optional) GT
    """
    gnss_path = os.path.join(INPUT_DIR, gnss_rel)
    imu_path = os.path.join(INPUT_DIR, imu_rel)

    # 1. Process Sensors
    gnss_df = process_gnss(gnss_path)
    if gnss_df is None or gnss_df.empty:
        return None

    imu_df = process_imu(imu_path)

    # 2. Merge GNSS and IMU
    # Create join key for GNSS
    gnss_df["merge_key"] = (gnss_df["UnixTimeMillis"] // 1000).astype(np.int64)

    if imu_df is not None and not imu_df.empty:
        merged_df = pd.merge(gnss_df, imu_df, on="merge_key", how="left")
    else:
        merged_df = gnss_df
        merged_df["accel_mag_mean"] = np.nan
        merged_df["gyro_mag_mean"] = np.nan

    # Fill missing IMU with mean or 0 (simple imputation)
    # Using 0 for magnitude implies no motion, which is a safe baseline assumption
    merged_df["accel_mag_mean"] = merged_df["accel_mag_mean"].fillna(9.8)  # Gravity
    merged_df["gyro_mag_mean"] = merged_df["gyro_mag_mean"].fillna(0.0)

    # 3. Add Metadata
    merged_df["drive_id"] = drive_id
    merged_df["phone_name"] = phone_name
    merged_df["tripId"] = f"{drive_id}-{phone_name}"

    # 4. Handle Ground Truth (Targets)
    if is_train and gt_rel:
        gt_path = os.path.join(INPUT_DIR, gt_rel)
        gt_df = pd.read_csv(gt_path)

        # Merge on exact timestamp
        final_df = pd.merge(
            merged_df,
            gt_df[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            on="UnixTimeMillis",
            how="inner",
        )

        if final_df.empty:
            return None

        # Compute Targets (ENU Residuals)
        # Convert WLS ECEF to LLA to use as reference for ENU conversion
        wls_lat, wls_lon, wls_alt = ecef_to_lla(
            final_df["WlsPositionXEcefMeters"].values,
            final_df["WlsPositionYEcefMeters"].values,
            final_df["WlsPositionZEcefMeters"].values,
        )

        # Convert GT LLA to ECEF
        gt_x, gt_y, gt_z = wgs84_to_ecef(
            final_df["LatitudeDegrees"].values,
            final_df["LongitudeDegrees"].values,
            np.zeros_like(
                final_df["LatitudeDegrees"]
            ),  # Altitude not critical for horiz error, assume 0 or use WLS alt
        )
        # Better: Use WLS altitude for GT ECEF conversion to isolate horizontal error?
        # Standard practice: Use 0 or ellipsoid surface.
        # Let's use WLS altitude to minimize vertical component mixing into horizontal.
        gt_x, gt_y, gt_z = wgs84_to_ecef(
            final_df["LatitudeDegrees"].values,
            final_df["LongitudeDegrees"].values,
            wls_alt,  # Use WLS altitude
        )

        # Compute ENU of GT relative to WLS
        # This gives the vector pointing FROM WLS TO GT
        d_east, d_north, d_up = ecef_to_enu(gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt)

        final_df["delta_east"] = d_east
        final_df["delta_north"] = d_north

        # Filter columns
        # Cite debug_lesson_6: Explicitly Whitelist Columns Required Downstream (Reconstruction/Metrics)
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        gt_cols = ["LatitudeDegrees", "LongitudeDegrees"]

        # Cite debug_lesson_11: Whitelist Non-Feature Metadata Required for Post-Processing
        keep_cols = (
            ["UnixTimeMillis", "drive_id", "phone_name", "tripId"]
            + FEATURE_COLS
            + TARGET_COLS
            + wls_cols
            + gt_cols
        )
        # Ensure all columns exist
        available_cols = [c for c in keep_cols if c in final_df.columns]
        return final_df[available_cols]

    else:
        # For Test/Inference
        # We need to return features + WLS position (for reconstruction)
        keep_cols = (
            ["UnixTimeMillis", "drive_id", "phone_name", "tripId"]
            + FEATURE_COLS
            + [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        )
        available_cols = [c for c in keep_cols if c in merged_df.columns]
        return merged_df[available_cols]


def generate_dataset(metadata_path, split_name, load_cached_data=True):
    """
    Main function to generate dataset for a split.
    """
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating {split_name} data from scratch...")
    meta_df = pd.read_csv(metadata_path)

    # Group by drive/phone to process files efficiently
    # meta_df has one row per timestamp, we need unique files

    cols = ["drive_id", "phone_name", "gnss_path", "imu_path"]
    if "gt_path" in meta_df.columns:
        cols.append("gt_path")

    unique_files = meta_df[cols].drop_duplicates()

    results = []

    for _, row in unique_files.iterrows():
        gt_rel = row["gt_path"] if "gt_path" in row else None

        df_drive = process_drive(
            row["drive_id"],
            row["phone_name"],
            row["gnss_path"],
            row["imu_path"],
            gt_rel,
            is_train=(split_name != "test"),
        )

        if df_drive is not None:
            # Filter to only rows requested in metadata (important for test/val alignment)
            # For train, we can use all available valid rows
            if split_name != "train":
                target_timestamps = meta_df[
                    (meta_df["drive_id"] == row["drive_id"])
                    & (meta_df["phone_name"] == row["phone_name"])
                ]["UnixTimeMillis"].values
                df_drive = df_drive[df_drive["UnixTimeMillis"].isin(target_timestamps)]

            results.append(df_drive)

    if not results:
        raise ValueError(f"No data generated for {split_name}")

    full_df = pd.concat(results, ignore_index=True)

    # Cite debug_lesson_10: Validate Schema Consistency Before Aggregating Data
    # Ensure no NaN targets in training/validation data
    if split_name != "test":
        original_len = len(full_df)
        full_df = full_df.dropna(subset=TARGET_COLS)
        dropped = original_len - len(full_df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN targets in {split_name} set.")

    # Save cache
    print(f"Saving {split_name} data to {cache_path}...")
    full_df.to_parquet(cache_path, index=False)

    return full_df
