import os
import numpy as np
import pandas as pd
from library.config import (
    OUTPUT_DIR,
    SEED,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    ML_FEATURES,
    TARGET_COLS,
)
from library.gnss_utils import lla2ecef, ecef2lla, ecef2enu, calculate_los_vector

# Constants
CLIGHT = 299792458.0


def compute_features_for_epoch(group):
    """
    Computes features for a single epoch (group of satellite measurements).
    """
    # 1. Receiver State Features (Take from first row as they are common for the epoch)
    # Handle potential NaNs in receiver state by filling with 0
    first_row = group.iloc[0]
    rec_state = {
        "BiasNanos": first_row.get("BiasNanos", 0.0),
        "BiasUncertaintyNanos": first_row.get("BiasUncertaintyNanos", 0.0),
        "DriftNanosPerSecond": first_row.get("DriftNanosPerSecond", 0.0),
        "DriftUncertaintyNanosPerSecond": first_row.get(
            "DriftUncertaintyNanosPerSecond", 0.0
        ),
    }

    # 2. Satellite Stats
    sat_stats = {
        "Cn0DbHz_mean": group["Cn0DbHz"].mean(),
        "Cn0DbHz_std": group["Cn0DbHz"].std(
            ddof=0
        ),  # ddof=0 to avoid NaN for single sat
        "SvElevationDegrees_mean": group["SvElevationDegrees"].mean(),
        "SvAzimuthDegrees_mean": group["SvAzimuthDegrees"].mean(),
        "SatCount": len(group),
    }

    # 3. Geometric Features (Unified Force & DOPs)
    # User Position (WLS Baseline)
    u_pos = first_row[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values.astype(float)

    # Satellite Positions
    s_pos = group[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values.astype(float)

    # Line of Sight Vectors
    # Vector from User to Sat
    diff = s_pos - u_pos
    dist = np.linalg.norm(diff, axis=1)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        los = diff / dist[:, None]
    los = np.nan_to_num(los)

    # Weights based on Signal Strength (Cn0)
    # Convert dBHz to linear scale approximation for weighting
    cn0 = group["Cn0DbHz"].values
    weights = 10 ** (0.1 * cn0)
    weights = weights / (np.sum(weights) + 1e-9)  # Normalize

    # Pseudorange Residuals
    # P_corr = P_raw - SatClk - Isrb - Iono - Tropo
    # Residual = P_corr - GeometricDist
    # Note: P_raw includes Receiver Clock Bias. The residual will thus be dominated by RxClkBias.
    # However, since we include RxClkBias features, the ML model can use this 'Force' vector
    # which points in the direction of the aggregate error (Position Error + Clock Error).

    p_raw = group["RawPseudorangeMeters"].values
    sat_clk = group["SvClockBiasMeters"].values
    isrb = group["IsrbMeters"].values
    iono = group["IonosphericDelayMeters"].values
    tropo = group["TroposphericDelayMeters"].values

    # Fill NaNs in corrections with 0
    sat_clk = np.nan_to_num(sat_clk)
    isrb = np.nan_to_num(isrb)
    iono = np.nan_to_num(iono)
    tropo = np.nan_to_num(tropo)

    p_corr = p_raw - sat_clk - isrb - iono - tropo
    residuals = p_corr - dist

    # Unified Geometric Force
    # Weighted sum of residuals projected onto LOS vectors
    # F = sum(w * res * u)
    force = np.sum(weights[:, None] * residuals[:, None] * los, axis=0)

    # Rotate Force to ENU
    # Need Reference LLA for rotation
    lat0, lon0, alt0 = ecef2lla(u_pos[0], u_pos[1], u_pos[2])
    f_e, f_n, f_u = ecef2enu(
        u_pos[0] + force[0], u_pos[1] + force[1], u_pos[2] + force[2], lat0, lon0, alt0
    )

    # Geometry Stiffness (DOPs)
    # H matrix: [LOS_x, LOS_y, LOS_z, 1]
    # We use weighted DOP: Q = inv(H.T * W * H)
    # Construct Weight Matrix
    W_diag = weights

    H = np.column_stack([los, np.ones(len(los))])

    try:
        # Weighted Least Squares Normal Matrix
        # J = H.T @ W @ H
        # We can just multiply H rows by sqrt(W) and do J = H_w.T @ H_w
        H_w = H * np.sqrt(W_diag)[:, None]
        J = H_w.T @ H_w

        Q = np.linalg.inv(J)

        # DOPs
        # GDOP = sqrt(trace(Q))
        # PDOP = sqrt(Qxx + Qyy + Qzz)
        # HDOP/VDOP require rotation to ENU.
        # For simplicity/robustness, we use ECEF DOPs or just PDOP.
        # Let's compute PDOP.
        pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])

        # To get HDOP/VDOP, we need to rotate Q to local frame.
        # Rot matrix R (3x3) from ECEF to ENU
        # Cov_enu = R * Q_xyz * R.T
        # Calculate R
        lat_rad, lon_rad = np.deg2rad(lat0), np.deg2rad(lon0)
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

        R = np.array(
            [
                [-sin_lon, cos_lon, 0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ]
        )

        Q_xyz = Q[:3, :3]
        Q_enu = R @ Q_xyz @ R.T

        hdop = np.sqrt(Q_enu[0, 0] + Q_enu[1, 1])
        vdop = np.sqrt(Q_enu[2, 2])

    except np.linalg.LinAlgError:
        pdop, hdop, vdop = 100.0, 100.0, 100.0

    geo_feats = {
        "Unified_Force_E": f_e,
        "Unified_Force_N": f_n,
        "Unified_Force_U": f_u,
        "HDOP": hdop,
        "VDOP": vdop,
        "PDOP": pdop,
    }

    # Combine all
    return {**rec_state, **sat_stats, **geo_feats}


def process_drive(drive_id, phone_name, gnss_path, gt_path=None, load_cached_data=True):
    """
    Processes a single drive to generate features and targets.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        gnss_path (str): Path to device_gnss.csv.
        gt_path (str, optional): Path to ground_truth.csv.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        pd.DataFrame: DataFrame with 'UnixTimeMillis', features, and targets (if gt_path provided).
    """
    # 1. Cache Setup
    cache_dir = os.path.join(OUTPUT_DIR, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"features_{drive_id}_{phone_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # 2. Load GNSS Data
    try:
        df_gnss = pd.read_csv(gnss_path)
    except FileNotFoundError:
        print(f"File not found: {gnss_path}")
        return pd.DataFrame()

    # 3. Clean Data
    # Drop rows where WLS baseline is missing
    df_gnss = df_gnss.dropna(
        subset=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    )

    if df_gnss.empty:
        return pd.DataFrame()

    # 4. Feature Engineering Loop
    # Group by epoch
    grouped = df_gnss.groupby("utcTimeMillis")

    feature_rows = []
    timestamps = []

    # Iterate over groups (epochs)
    for ts, group in grouped:
        feats = compute_features_for_epoch(group)
        feature_rows.append(feats)
        timestamps.append(ts)

    df_features = pd.DataFrame(feature_rows)
    df_features["UnixTimeMillis"] = timestamps

    # 5. Target Generation (if GT provided)
    if gt_path and os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)

        # Merge features with GT on timestamp
        df_merged = pd.merge(
            df_features,
            df_gt[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            on="UnixTimeMillis",
            how="inner",
        )

        if not df_merged.empty:
            # Get WLS positions for the merged timestamps
            # We need to get WLS positions from the original GNSS dataframe corresponding to these timestamps
            # Since WLS is same for all sats in epoch, we take the first one
            wls_pos = df_gnss.drop_duplicates("utcTimeMillis")[
                [
                    "utcTimeMillis",
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ]

            df_merged = pd.merge(
                df_merged,
                wls_pos,
                left_on="UnixTimeMillis",
                right_on="utcTimeMillis",
                how="left",
            )

            # --- Altitude Correction Logic ---
            # 1. Convert WLS ECEF to LLA
            wls_x = df_merged["WlsPositionXEcefMeters"].values
            wls_y = df_merged["WlsPositionYEcefMeters"].values
            wls_z = df_merged["WlsPositionZEcefMeters"].values

            wls_lat, wls_lon, wls_alt = ecef2lla(wls_x, wls_y, wls_z)

            # 2. Construct Corrected GT LLA (GT Lat/Lon, WLS Alt)
            gt_lat = df_merged["LatitudeDegrees"].values
            gt_lon = df_merged["LongitudeDegrees"].values

            # 3. Convert Corrected GT to ECEF
            gt_corr_x, gt_corr_y, gt_corr_z = lla2ecef(gt_lat, gt_lon, wls_alt)

            # 4. Convert Corrected GT ECEF to ENU relative to WLS ECEF (Reference)
            # This gives the residuals we want to predict
            res_e, res_n, res_u = ecef2enu(
                gt_corr_x, gt_corr_y, gt_corr_z, wls_lat, wls_lon, wls_alt
            )

            df_merged["res_E"] = res_e
            df_merged["res_N"] = res_n
            # res_U should be effectively 0, but we don't need it as target

            # Cleanup
            df_final = df_merged.drop(
                columns=[
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                    "utcTimeMillis",
                ]
            )
        else:
            df_final = pd.DataFrame()

    else:
        # Test mode or no GT
        df_final = df_features

        # We still need WLS positions for the inference stage (to add predictions back)
        # Add WLS columns to the feature dataframe for convenience
        wls_pos = df_gnss.drop_duplicates("utcTimeMillis")[
            [
                "utcTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        df_final = pd.merge(
            df_final,
            wls_pos,
            left_on="UnixTimeMillis",
            right_on="utcTimeMillis",
            how="left",
        )
        df_final = df_final.drop(columns=["utcTimeMillis"])

    # 6. Save to Cache
    if not df_final.empty:
        df_final.to_parquet(cache_file)

    return df_final
