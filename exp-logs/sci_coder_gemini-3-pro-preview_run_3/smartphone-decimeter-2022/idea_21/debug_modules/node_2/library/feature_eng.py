import os
import numpy as np
import pandas as pd
from library.config import WORKING_DIR, L1_FREQ, L5_FREQ, LIGHT_SPEED, WGS84_A, WGS84_F
from library.utils import ecef_to_enu, wgs84_to_ecef, ecef_to_wgs84
from library.data_loader import load_drive_data


def compute_satellite_residuals(gnss_df):
    """
    Calculate Pseudorange and Doppler residuals relative to WLS baseline.
    """
    # Ensure we have WLS positions
    df = gnss_df.dropna(
        subset=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
        ]
    ).copy()

    # --- 1. Compute Geometric Range ---
    # Vector from WLS to Sat
    dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
    dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
    dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]

    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Line of Sight Unit Vectors (ECEF)
    df["los_x"] = dx / dist
    df["los_y"] = dy / dist
    df["los_z"] = dz / dist

    # --- 2. Pseudorange Residuals ---
    # Corrected PR = Raw - SatClk - ISRB - Iono - Tropo
    # Note: RawPseudorange usually includes RxClkBias.
    # We want Residual = CorrectedPR - GeometricRange ~ RxClkBias + PosError

    # Fill missing corrections with 0
    df["SvClockBiasMeters"] = df["SvClockBiasMeters"].fillna(0)
    df["IsrbMeters"] = df["IsrbMeters"].fillna(0)
    df["IonosphericDelayMeters"] = df["IonosphericDelayMeters"].fillna(0)
    df["TroposphericDelayMeters"] = df["TroposphericDelayMeters"].fillna(0)

    corrected_pr = (
        df["RawPseudorangeMeters"]
        + df["SvClockBiasMeters"]
        - df["IsrbMeters"]
        - df["IonosphericDelayMeters"]
        - df["TroposphericDelayMeters"]
    )

    raw_pr_residual = corrected_pr - dist

    # Remove Receiver Clock Bias (Median per epoch)
    # This isolates the geometric/multipath error component
    epoch_bias = raw_pr_residual.groupby(df["UnixTimeMillis"]).transform("median")
    df["pr_residual"] = raw_pr_residual - epoch_bias

    # --- 3. Doppler Residuals ---
    # Expected Range Rate = LOS dot (SatVel - RxVel)
    # Assuming RxVel is small or captured by drift, we project SatVel onto LOS
    # SatVel is usually provided. If not, we skip.

    if "SvVelocityXEcefMetersPerSecond" in df.columns:
        # Fill missing velocities
        df["SvVelocityXEcefMetersPerSecond"] = df[
            "SvVelocityXEcefMetersPerSecond"
        ].fillna(0)
        df["SvVelocityYEcefMetersPerSecond"] = df[
            "SvVelocityYEcefMetersPerSecond"
        ].fillna(0)
        df["SvVelocityZEcefMetersPerSecond"] = df[
            "SvVelocityZEcefMetersPerSecond"
        ].fillna(0)

        # Projected Satellite Velocity (Range Rate contribution from Sat)
        sat_range_rate = (
            df["SvVelocityXEcefMetersPerSecond"] * df["los_x"]
            + df["SvVelocityYEcefMetersPerSecond"] * df["los_y"]
            + df["SvVelocityZEcefMetersPerSecond"] * df["los_z"]
        )

        # Measured Pseudorange Rate (includes Clock Drift)
        # Residual = Measured - Expected
        raw_dop_residual = df["PseudorangeRateMetersPerSecond"] - sat_range_rate

        # Remove Clock Drift (Median per epoch)
        epoch_drift = raw_dop_residual.groupby(df["UnixTimeMillis"]).transform("median")
        df["dop_residual"] = raw_dop_residual - epoch_drift
    else:
        df["dop_residual"] = 0.0

    return df


def project_forces(gnss_df):
    """
    Project residuals into ENU frame and aggregate by band.
    """
    # Convert WLS ECEF to Lat/Lon for rotation matrix
    # We take the mean WLS position per epoch to define the local tangent plane
    # Or simply convert per-row. Since WLS is per-epoch, it's the same.
    # However, ecef_to_wgs84 is computationally expensive to run per row if not vectorized.
    # We'll assume the WLS position provided is the linearization point.

    # Vectorized ECEF to Geodetic (Lat/Lon) for rotation
    # Simplified version or use library if vectorized.
    # The library function is scalar/numpy array compatible.
    wls_x = gnss_df["WlsPositionXEcefMeters"].values
    wls_y = gnss_df["WlsPositionYEcefMeters"].values
    wls_z = gnss_df["WlsPositionZEcefMeters"].values

    lat, lon, _ = ecef_to_wgs84(wls_x, wls_y, wls_z)

    # Convert Lat/Lon to Radians
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotate LOS vectors (ECEF) to ENU
    # los_e = -sin_lam * los_x + cos_lam * los_y
    # los_n = -sin_phi * cos_lam * los_x - sin_phi * sin_lam * los_y + cos_phi * los_z
    # los_u = cos_phi * cos_lam * los_x + cos_phi * sin_lam * los_y + sin_phi * los_z

    los_x = gnss_df["los_x"].values
    los_y = gnss_df["los_y"].values
    los_z = gnss_df["los_z"].values

    los_e = -sin_lam * los_x + cos_lam * los_y
    los_n = -sin_phi * cos_lam * los_x - sin_phi * sin_lam * los_y + cos_phi * los_z
    los_u = cos_phi * cos_lam * los_x + cos_phi * sin_lam * los_y + sin_phi * los_z

    gnss_df["los_e"] = los_e
    gnss_df["los_n"] = los_n
    gnss_df["los_u"] = los_u

    # Define Weights (Inverse Variance)
    # Avoid division by zero
    unc = gnss_df["RawPseudorangeUncertaintyMeters"].fillna(10.0)
    gnss_df["weight"] = 1.0 / (unc**2 + 1e-6)

    # Identify Bands
    # L5/E5a ~ 1.176 GHz. L1 ~ 1.575 GHz. Cutoff at 1.3 GHz.
    # If CarrierFreq is missing, assume L1 (most common).
    freq = gnss_df["CarrierFrequencyHz"].fillna(L1_FREQ)
    is_l5 = freq < 1.3e9

    # Pre-compute weighted force components
    # Force = Residual * LOS * Weight

    # L1 Pseudorange Forces
    mask_l1 = ~is_l5
    gnss_df["F_L1_E"] = np.where(
        mask_l1, gnss_df["pr_residual"] * gnss_df["los_e"] * gnss_df["weight"], 0
    )
    gnss_df["F_L1_N"] = np.where(
        mask_l1, gnss_df["pr_residual"] * gnss_df["los_n"] * gnss_df["weight"], 0
    )
    gnss_df["F_L1_U"] = np.where(
        mask_l1, gnss_df["pr_residual"] * gnss_df["los_u"] * gnss_df["weight"], 0
    )
    gnss_df["W_L1"] = np.where(mask_l1, gnss_df["weight"], 0)

    # L5 Pseudorange Forces
    mask_l5 = is_l5
    gnss_df["F_L5_E"] = np.where(
        mask_l5, gnss_df["pr_residual"] * gnss_df["los_e"] * gnss_df["weight"], 0
    )
    gnss_df["F_L5_N"] = np.where(
        mask_l5, gnss_df["pr_residual"] * gnss_df["los_n"] * gnss_df["weight"], 0
    )
    gnss_df["F_L5_U"] = np.where(
        mask_l5, gnss_df["pr_residual"] * gnss_df["los_u"] * gnss_df["weight"], 0
    )
    gnss_df["W_L5"] = np.where(mask_l5, gnss_df["weight"], 0)

    # Doppler Forces (All bands aggregated, usually L1 dominates count)
    # Weight for Doppler: use PseudorangeRateUncertaintyMetersPerSecond
    dop_unc = gnss_df["PseudorangeRateUncertaintyMetersPerSecond"].fillna(1.0)
    w_dop = 1.0 / (dop_unc**2 + 1e-6)

    gnss_df["F_Dop_E"] = gnss_df["dop_residual"] * gnss_df["los_e"] * w_dop
    gnss_df["F_Dop_N"] = gnss_df["dop_residual"] * gnss_df["los_n"] * w_dop
    gnss_df["F_Dop_U"] = gnss_df["dop_residual"] * gnss_df["los_u"] * w_dop
    gnss_df["W_Dop"] = w_dop

    # Aggregation
    agg_funcs = {
        "F_L1_E": "sum",
        "F_L1_N": "sum",
        "F_L1_U": "sum",
        "W_L1": "sum",
        "F_L5_E": "sum",
        "F_L5_N": "sum",
        "F_L5_U": "sum",
        "W_L5": "sum",
        "F_Dop_E": "sum",
        "F_Dop_N": "sum",
        "F_Dop_U": "sum",
        "W_Dop": "sum",
        "Cn0DbHz": "mean",
        "Svid": "count",  # Total SV count
        "WlsPositionXEcefMeters": "first",  # Anchor position
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Include GT columns if they exist (for training)
    gt_cols = ["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]
    for col in gt_cols:
        if col in gnss_df.columns:
            agg_funcs[col] = "first"

    # Include Context Identifiers (Cite debug_lesson_15, debug_lesson_17)
    for col in ["tripId", "drive_id", "phone_name"]:
        if col in gnss_df.columns:
            agg_funcs[col] = "first"

    grouped = gnss_df.groupby("UnixTimeMillis").agg(agg_funcs)

    # Normalize Forces by Total Weight (to get weighted average residual vector)
    # Avoid division by zero
    for band in ["L1", "L5", "Dop"]:
        w = grouped[f"W_{band}"]
        # If weight is 0, force is 0
        grouped[f"F_{band}_E"] = np.where(w > 0, grouped[f"F_{band}_E"] / w, 0)
        grouped[f"F_{band}_N"] = np.where(w > 0, grouped[f"F_{band}_N"] / w, 0)
        grouped[f"F_{band}_U"] = np.where(w > 0, grouped[f"F_{band}_U"] / w, 0)

    # Add L5 Count specifically
    l5_counts = gnss_df[is_l5].groupby("UnixTimeMillis")["Svid"].count()
    grouped["L5_Count"] = l5_counts
    grouped["L5_Count"] = grouped["L5_Count"].fillna(0)

    grouped.reset_index(inplace=True)

    return grouped


def generate_features(drive_id, phone_name, metadata_df, load_cached_data=True):
    """
    Generate features for a specific drive.
    """
    # Cache path
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_file = f"features_{drive_id}-{phone_name}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    # Load Data
    gnss_df, _ = load_drive_data(
        drive_id, phone_name, metadata_df, load_cached_data=load_cached_data
    )

    if gnss_df.empty:
        return pd.DataFrame()

    # 1. Compute Residuals
    gnss_df = compute_satellite_residuals(gnss_df)

    # 2. Project Forces
    features_df = project_forces(gnss_df)

    # 3. Compute Targets (if GT available)
    if (
        "LatitudeDegrees" in features_df.columns
        and "LongitudeDegrees" in features_df.columns
    ):
        # Convert GT Lat/Lon to ECEF
        gt_x, gt_y, gt_z = wgs84_to_ecef(
            features_df["LatitudeDegrees"],
            features_df["LongitudeDegrees"],
            features_df["AltitudeMeters"].fillna(0),
        )

        # Get WLS Lat/Lon for ENU origin
        wls_x = features_df["WlsPositionXEcefMeters"]
        wls_y = features_df["WlsPositionYEcefMeters"]
        wls_z = features_df["WlsPositionZEcefMeters"]
        wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

        # Convert GT ECEF to ENU relative to WLS
        # We need to do this row-by-row or vectorized.
        # Since ecef_to_enu in utils is scalar/array based but expects scalar ref,
        # we need a vectorized version for varying reference.

        # Vectorized ENU conversion
        dx = gt_x - wls_x
        dy = gt_y - wls_y
        dz = gt_z - wls_z

        phi = np.deg2rad(wls_lat)
        lam = np.deg2rad(wls_lon)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)

        t_e = -sin_lam * dx + cos_lam * dy
        t_n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
        t_u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

        features_df["Target_E"] = t_e
        features_df["Target_N"] = t_n
        features_df["Target_U"] = t_u

    # Save cache
    features_df.to_parquet(cache_path, index=False)

    return features_df


def create_dataset(metadata_df, load_cached_data=True):
    """
    Iterate over metadata to create full dataset.
    """
    all_features = []

    # Group by drive and phone to process sequentially
    drive_phones = metadata_df[["drive_id", "phone_name"]].drop_duplicates().values

    for drive_id, phone_name in drive_phones:
        try:
            df = generate_features(
                drive_id, phone_name, metadata_df, load_cached_data=load_cached_data
            )
            if not df.empty:
                all_features.append(df)
        except Exception as e:
            print(f"Error processing {drive_id} {phone_name}: {e}")

    if not all_features:
        return pd.DataFrame()

    return pd.concat(all_features, ignore_index=True)
