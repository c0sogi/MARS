import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import llh_to_ecef, ecef_to_llh, ecef_to_enu
from library.data_io import load_drive_data


def compute_geometric_features(
    drive_id: str,
    phone_name: str,
    gnss_path: str,
    imu_path: str,
    load_cached_data: bool = True,
) -> pd.DataFrame:
    """
    Computes point-wise geometric features for ML 'Anchor' prediction.
    Features include Net Pseudorange Force and Geometry Stiffness in ENU frame.

    Args:
        drive_id: Drive identifier.
        phone_name: Phone model name.
        gnss_path: Relative path to GNSS data.
        imu_path: Relative path to IMU data.
        load_cached_data: Whether to load from cache if available.

    Returns:
        DataFrame with columns ['UnixTimeMillis', 'NetForce_E', 'NetForce_N', ...].
    """
    # 1. Cache Check
    cache_dir = os.path.join(Config.WORKING_DIR, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Sanitize filenames
    safe_drive = drive_id.replace("/", "_").replace("\\", "_")
    safe_phone = phone_name.replace("/", "_").replace("\\", "_")
    cache_path = os.path.join(cache_dir, f"features_{safe_drive}_{safe_phone}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Warning: Failed to load feature cache: {e}. Recomputing...")

    # 2. Load Data
    data = load_drive_data(
        drive_id, phone_name, gnss_path, imu_path, gt_path=None, load_cached_data=True
    )
    df_gnss = data["gnss"]
    df_imu = data["imu"]

    if df_gnss.empty:
        return pd.DataFrame()

    # 3. Preprocessing GNSS
    # Filter valid signals based on Cn0 and Elevation
    # We use all bands (L1/L5) as requested
    mask = (
        (df_gnss["Cn0DbHz"] >= Config.CN0_THRESHOLD)
        & (df_gnss["SvElevationDegrees"] >= Config.ELEVATION_MASK)
        & (df_gnss["WlsPositionXEcefMeters"].notna())
    )

    df_clean = df_gnss[mask].copy()

    if df_clean.empty:
        return pd.DataFrame()

    # 4. Compute Satellite Geometry vectors (Vectorized)
    # User WLS Position (ECEF)
    u_x = df_clean["WlsPositionXEcefMeters"].values
    u_y = df_clean["WlsPositionYEcefMeters"].values
    u_z = df_clean["WlsPositionZEcefMeters"].values

    # Satellite Position (ECEF)
    s_x = df_clean["SvPositionXEcefMeters"].values
    s_y = df_clean["SvPositionYEcefMeters"].values
    s_z = df_clean["SvPositionZEcefMeters"].values

    # Line of Sight Vector (ECEF): Sat - User
    dx = s_x - u_x
    dy = s_y - u_y
    dz = s_z - u_z
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Normalize to get Unit vectors (ECEF)
    # Handle division by zero if any (unlikely given elevation mask)
    dist = np.where(dist < 1e-3, 1e-3, dist)
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    # 5. Compute Residuals
    # RawPseudorangeMeters contains: Distance + ClockBias + Errors
    # We estimate a common ClockBias per epoch as median(Pseudorange - GeometricDistance)
    df_clean["geom_dist"] = dist
    df_clean["raw_res"] = df_clean["RawPseudorangeMeters"] - df_clean["geom_dist"]

    # Calculate epoch-level clock bias estimate
    epoch_bias = df_clean.groupby("utcTimeMillis")["raw_res"].transform("median")

    # Residual r_i: The "excess" distance not explained by geometry or common clock
    residuals = df_clean["raw_res"] - epoch_bias

    # Weights: Based on Signal Strength
    weights = 10 ** (df_clean["Cn0DbHz"] / 10.0)

    # 6. Rotate to ENU Frame
    # We need rotation matrix for each point based on its WLS position.
    # We approximate the rotation matrix calculation using a spherical earth model for speed,
    # or use the vectorized ellipsoid parameters.

    # WGS84 Constants
    a = 6378137.0
    e2 = 6.69437999014e-3

    # Vectorized Lat/Lon calculation from ECEF
    p = np.sqrt(u_x**2 + u_y**2)
    lon = np.arctan2(u_y, u_x)
    # Initial lat approximation
    lat = np.arctan2(u_z, p * (1 - e2))
    # One iteration for better precision
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    alt = p / np.cos(lat) - N
    lat = np.arctan2(u_z, p * (1 - e2 * (N / (N + alt))))

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Project Unit Vectors to ENU
    # Rotation Matrix R_ecef2enu rows:
    # East:  [-sin_lon, cos_lon, 0]
    # North: [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
    # Up:    [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]

    u_e = -sin_lon * ux + cos_lon * uy
    u_n = -sin_lat * cos_lon * ux - sin_lat * sin_lon * uy + cos_lat * uz
    u_u = cos_lat * cos_lon * ux + cos_lat * sin_lon * uy + sin_lat * uz

    # 7. Compute Feature Terms
    # Weighted Force Components: F = w * r * u
    # Note: If residual is positive (measured > geom), it implies satellite is "pushing"
    # the solution away. The correction should "pull" it back.
    # We let the ML model learn the sign convention.
    w_r = weights * residuals
    df_clean["F_e"] = w_r * u_e
    df_clean["F_n"] = w_r * u_n
    df_clean["F_u"] = w_r * u_u

    # Geometry Matrix Elements (Weighted Stiffness): G = w * u * u^T
    df_clean["G_ee"] = weights * u_e * u_e
    df_clean["G_nn"] = weights * u_n * u_n
    df_clean["G_uu"] = weights * u_u * u_u
    df_clean["G_en"] = weights * u_e * u_n
    df_clean["G_eu"] = weights * u_e * u_u
    df_clean["G_nu"] = weights * u_n * u_u

    # 8. Aggregation
    agg_funcs = {
        "F_e": "sum",
        "F_n": "sum",
        "F_u": "sum",
        "G_ee": "sum",
        "G_nn": "sum",
        "G_uu": "sum",
        "G_en": "sum",
        "G_eu": "sum",
        "G_nu": "sum",
        "Cn0DbHz": ["mean", "std", "max"],
        "Svid": "count",
    }

    df_features = df_clean.groupby("utcTimeMillis").agg(agg_funcs)

    # Flatten MultiIndex columns
    df_features.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_features.columns.values
    ]

    # Rename for clarity
    df_features = df_features.rename(
        columns={
            "F_e_sum": "NetForce_E",
            "F_n_sum": "NetForce_N",
            "F_u_sum": "NetForce_U",
            "Svid_count": "SatCount",
        }
    )

    df_features.reset_index(inplace=True)
    df_features.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # 9. IMU Aggregation (if available)
    if not df_imu.empty:
        # Calculate magnitude
        df_imu["mag"] = np.sqrt(
            df_imu["MeasurementX"] ** 2
            + df_imu["MeasurementY"] ** 2
            + df_imu["MeasurementZ"] ** 2
        )

        # Filter for Accelerometer
        accel = df_imu[df_imu["MessageType"] == "UncalAccel"].copy()
        if not accel.empty:
            # Align timestamps: Round IMU time to nearest second (1000ms)
            # GNSS timestamps are typically exact seconds (e.g., 1234567890000)
            accel["UnixTimeMillis"] = np.round(accel["utcTimeMillis"] / 1000.0) * 1000.0

            accel_agg = (
                accel.groupby("UnixTimeMillis")["mag"]
                .agg(["mean", "std"])
                .reset_index()
            )
            accel_agg.columns = ["UnixTimeMillis", "Accel_Mean", "Accel_Std"]

            # Merge with GNSS features
            df_features = pd.merge(
                df_features, accel_agg, on="UnixTimeMillis", how="left"
            )

    # Fill missing IMU data (if gaps exist)
    if "Accel_Mean" in df_features.columns:
        df_features["Accel_Mean"] = df_features["Accel_Mean"].fillna(
            9.8
        )  # Standard gravity
        df_features["Accel_Std"] = df_features["Accel_Std"].fillna(0.0)
    else:
        df_features["Accel_Mean"] = 9.8
        df_features["Accel_Std"] = 0.0

    # 10. Save to Cache
    try:
        df_features.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save feature cache: {e}")

    return df_features
