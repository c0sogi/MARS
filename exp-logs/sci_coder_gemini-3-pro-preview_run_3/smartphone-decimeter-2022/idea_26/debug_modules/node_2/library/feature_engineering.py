import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import (
    WORKING_DIR,
    ML_FEATURES,
    TARGET_E,
    TARGET_N,
    SEED,
    CLLIGHT,
)
from library.utils import ecef_to_geodetic, geodetic_to_ecef, ecef_to_enu
from library.data_loader import load_metadata, load_gnss_dataframe, load_ground_truth

# Ensure working directory exists
os.makedirs(os.path.join(WORKING_DIR, "features_cache"), exist_ok=True)


def calculate_satellite_geometry(group):
    """
    Computes geometric features for a single epoch (group of satellite measurements).
    Features:
    - Geometric Force (x, y, z): Weighted sum of unit vectors scaled by residuals.
    - Stiffness (x, y, z): Diagonal elements of the DOP matrix (H^T H)^-1.
    """
    # Extract necessary columns
    sv_x = group["SvPositionXEcefMeters"].values
    sv_y = group["SvPositionYEcefMeters"].values
    sv_z = group["SvPositionZEcefMeters"].values

    wls_x = group["WlsPositionXEcefMeters"].values[0]
    wls_y = group["WlsPositionYEcefMeters"].values[0]
    wls_z = group["WlsPositionZEcefMeters"].values[0]

    raw_pr = group["RawPseudorangeMeters"].values
    sv_clk = group["SvClockBiasMeters"].values

    # Signal strength weights (Cn0)
    cn0 = group["Cn0DbHz"].values
    # Convert dBHz to linear scale for weighting: w ~ 10^(Cn0/10)
    weights = 10 ** (cn0 / 10.0)
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones_like(weights) / len(weights)

    # 1. Compute Unit Vectors (Line of Sight)
    dx = sv_x - wls_x
    dy = sv_y - wls_y
    dz = sv_z - wls_z
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Handle potential division by zero (unlikely)
    dist = np.where(dist < 1e-3, 1e-3, dist)

    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    # 2. Compute Pseudorange Residuals
    # Residual = Observed - Computed
    # Observed = RawPr - SvClk
    # Computed = GeometricDist + RxClkBias (RxClkBias is unknown/common)
    # Pre-residual = RawPr - SvClk - GeometricDist
    pre_residual = raw_pr - sv_clk - dist

    # Remove Receiver Clock Bias (Common Mode) using Median
    # This leaves the geometric error + noise
    rx_clk_bias_est = np.median(pre_residual)
    residual = pre_residual - rx_clk_bias_est

    # 3. Geometric Force
    # Gradient of the error surface
    # F = Sum(w_i * res_i * u_i)
    force_x = np.sum(weights * residual * ux)
    force_y = np.sum(weights * residual * uy)
    force_z = np.sum(weights * residual * uz)

    # 4. Geometry Stiffness (DOP approximation)
    # Design Matrix H = [ux, uy, uz, 1]
    # We compute diagonal of (H^T H)^-1
    # We use unweighted DOP for pure geometry stiffness
    try:
        ones = np.ones_like(ux)
        H = np.column_stack((ux, uy, uz, ones))

        # H^T H
        HtH = H.T @ H

        # Inverse
        Q = np.linalg.inv(HtH)

        stiffness_x = Q[0, 0]
        stiffness_y = Q[1, 1]
        stiffness_z = Q[2, 2]
    except np.linalg.LinAlgError:
        # Fallback for singular matrix (poor geometry)
        stiffness_x = 100.0
        stiffness_y = 100.0
        stiffness_z = 100.0

    return pd.Series(
        {
            "force_x": force_x,
            "force_y": force_y,
            "force_z": force_z,
            "stiffness_x": stiffness_x,
            "stiffness_y": stiffness_y,
            "stiffness_z": stiffness_z,
        }
    )


def extract_receiver_state(group):
    """
    Extracts receiver clock state and signal aggregates for an epoch.
    Assumes these values are constant or we take the first valid one per epoch.
    """
    first = group.iloc[0]

    return pd.Series(
        {
            "BiasNanos": first.get("BiasNanos", 0),
            "BiasUncertaintyNanos": first.get("BiasUncertaintyNanos", 0),
            "DriftNanosPerSecond": first.get("DriftNanosPerSecond", 0),
            "DriftUncertaintyNanosPerSecond": first.get(
                "DriftUncertaintyNanosPerSecond", 0
            ),
            "Cn0DbHz_mean": group["Cn0DbHz"].mean(),
            "sv_count": len(group),
        }
    )


def compute_features_from_gnss(gnss_df):
    """
    Main feature engineering function.
    Aggregates GNSS data by (tripId, UnixTimeMillis) to create ML features.
    """
    print("Computing geometric and state features...")

    # Group by Epoch
    # We use tripId and utcTimeMillis as unique identifier for an epoch
    grouped = gnss_df.groupby(["tripId", "utcTimeMillis"])

    # Apply geometric calculations
    # This can be slow, so we use tqdm with pandas apply if possible, or iterate
    # For performance on large datasets, we iterate chunks or use parallelization if available.
    # Here we stick to simple apply for clarity and robustness within single file constraint.

    # 1. Geometric Features
    geom_features = grouped.apply(calculate_satellite_geometry)

    # 2. State Features
    state_features = grouped.apply(extract_receiver_state)

    # Combine
    features = pd.concat([geom_features, state_features], axis=1).reset_index()

    # Rename utcTimeMillis to UnixTimeMillis to match metadata/GT
    features.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    return features


def compute_altitude_corrected_target(merged_df):
    """
    Computes the Altitude-Corrected ENU residuals.

    Logic:
    1. Get WLS Position (Baseline) -> Convert to Geodetic (Lat_wls, Lon_wls, Alt_wls).
    2. Get Ground Truth (Lat_gt, Lon_gt).
    3. Project GT to WLS Altitude: P_proj_geo = (Lat_gt, Lon_gt, Alt_wls).
    4. Convert P_proj_geo to ECEF -> P_proj_ecef.
    5. Calculate Vector: V = P_proj_ecef - WLS_ecef.
    6. Rotate V to ENU frame centered at WLS.
    7. Target = (E, N).
    """
    # Check if GT columns exist
    if (
        "LatitudeDegrees" not in merged_df.columns
        or "LongitudeDegrees" not in merged_df.columns
    ):
        return merged_df

    # We need WLS position. In the aggregated features, we don't have WLS columns directly.
    # We need to retrieve them. However, WLS position is per-epoch.
    # The load_gnss_dataframe function returns raw rows.
    # We need to preserve WLS position during feature aggregation or merge it back.
    # Let's assume we merge it back from the raw GNSS (taking the first row per epoch).

    # Vectorized computation

    # 1. WLS ECEF
    wls_x = merged_df["WlsPositionXEcefMeters"]
    wls_y = merged_df["WlsPositionYEcefMeters"]
    wls_z = merged_df["WlsPositionZEcefMeters"]

    # 2. Convert WLS to Geodetic to get Alt_wls and Reference Lat/Lon for ENU rotation
    wls_lat, wls_lon, wls_alt = ecef_to_geodetic(
        wls_x.values, wls_y.values, wls_z.values
    )

    # 3. Ground Truth
    gt_lat = merged_df["LatitudeDegrees"].values
    gt_lon = merged_df["LongitudeDegrees"].values

    # 4. Project GT to WLS Altitude
    # P_proj_geo = (gt_lat, gt_lon, wls_alt)
    proj_x, proj_y, proj_z = geodetic_to_ecef(gt_lat, gt_lon, wls_alt)

    # 5. Calculate Difference Vector in ECEF
    dx = proj_x - wls_x
    dy = proj_y - wls_y
    dz = proj_z - wls_z

    # 6. Rotate to ENU (centered at WLS Lat/Lon)
    # Note: We rotate relative to WLS position because that's what the model sees as "local origin"
    dE, dN, dU = ecef_to_enu(proj_x, proj_y, proj_z, wls_lat, wls_lon, wls_alt)

    # Assign targets
    merged_df[TARGET_E] = dE
    merged_df[TARGET_N] = dN

    # Sanity check: dU should be close to 0
    # merged_df["dU_check"] = dU

    return merged_df


def generate_dataset(split_name: str, load_cached_data: bool = True):
    """
    Orchestrates the generation of the dataset (Features + Targets) for a given split.
    Handles caching.
    """
    cache_file = os.path.join(
        WORKING_DIR, "features_cache", f"dataset_{split_name}.parquet"
    )

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading dataset for '{split_name}' from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Generating dataset for '{split_name}'...")

    # 2. Load Metadata
    meta_df = load_metadata(split_name)

    # 3. Load GNSS Data
    gnss_df = load_gnss_dataframe(
        meta_df, split_name, load_cached_data=load_cached_data
    )

    # 4. Compute Features
    features_df = compute_features_from_gnss(gnss_df)

    # 5. Merge with Metadata (to get GT for train/val, and trip info)
    # Metadata has (tripId, UnixTimeMillis) as key
    dataset_df = pd.merge(
        meta_df, features_df, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    # 6. Add WLS Position to dataset_df for Target Calculation
    # We need WLS position to compute the Altitude-Corrected Target.
    # Extract WLS from GNSS (one per epoch)
    wls_df = (
        gnss_df.groupby(["tripId", "utcTimeMillis"])
        .first()[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .reset_index()
    )
    wls_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    dataset_df = pd.merge(
        dataset_df, wls_df, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    # 7. Compute Targets (only for train/val)
    if split_name in ["train", "val"]:
        dataset_df = compute_altitude_corrected_target(dataset_df)

        # Drop rows where targets are NaN (if any)
        dataset_df = dataset_df.dropna(subset=[TARGET_E, TARGET_N])

    # 8. Save Cache
    print(f"Saving dataset for '{split_name}' to cache: {cache_file}")
    dataset_df.to_parquet(cache_file, index=False)

    return dataset_df
