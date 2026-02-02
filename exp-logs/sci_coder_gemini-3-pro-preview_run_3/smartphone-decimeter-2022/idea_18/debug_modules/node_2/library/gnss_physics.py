import os
import numpy as np
import pandas as pd
from library.coordinate_utils import WGS84_to_ECEF, ECEF_to_ENU, ECEF_to_WGS84

# Constants
CLIGHT = 299792458.0
CACHE_DIR = "./working/idea_18"


def apply_physics_transformations(gnss_df):
    """
    Computes per-satellite physical residuals and Line-of-Sight (LOS) vectors.

    Args:
        gnss_df (pd.DataFrame): Cleaned GNSS dataframe.

    Returns:
        pd.DataFrame: Dataframe with added 'res_pr', 'res_dop', and 'los_*' columns.
    """
    df = gnss_df.copy()

    # --- 1. Compute Geometric Range and LOS Vectors (ECEF) ---
    # User Position (WLS)
    u_x = df["WlsPositionXEcefMeters"].values
    u_y = df["WlsPositionYEcefMeters"].values
    u_z = df["WlsPositionZEcefMeters"].values

    # Satellite Position
    s_x = df["SvPositionXEcefMeters"].values
    s_y = df["SvPositionYEcefMeters"].values
    s_z = df["SvPositionZEcefMeters"].values

    # Vector from User to Satellite
    dx = s_x - u_x
    dy = s_y - u_y
    dz = s_z - u_z

    # Geometric Range
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # LOS Unit Vectors
    # Handle division by zero if r is 0 (unlikely but safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        los_x = dx / r
        los_y = dy / r
        los_z = dz / r

    # --- 2. Compute Pseudorange Residuals ---
    # Fill NaNs in correction terms with 0
    df["SvClockBiasMeters"] = df["SvClockBiasMeters"].fillna(0)
    df["IsrbMeters"] = df["IsrbMeters"].fillna(0)
    df["IonosphericDelayMeters"] = df["IonosphericDelayMeters"].fillna(0)
    df["TroposphericDelayMeters"] = df["TroposphericDelayMeters"].fillna(0)

    # Calculate Corrected Pseudorange
    # PR = Range + c*(dt_u - dt_s) + I + T + Errors
    # Corrected PR removes known terms: PR + dt_s - I - T - ISRB
    corrected_pr = (
        df["RawPseudorangeMeters"]
        + df["SvClockBiasMeters"]
        - df["IsrbMeters"]
        - df["IonosphericDelayMeters"]
        - df["TroposphericDelayMeters"]
    )

    # Raw Residual = Corrected PR - Geometric Range
    # This residual contains the Receiver Clock Bias (c * dt_u) + Position Error + Multipath
    raw_res_pr = corrected_pr - r

    # Remove Receiver Clock Bias (Common Mode Error) per epoch
    # We assume the median residual approximates the clock bias
    epoch_bias_pr = raw_res_pr.groupby(df["utcTimeMillis"]).transform("median")
    res_pr = raw_res_pr - epoch_bias_pr

    # --- 3. Compute Doppler Residuals ---
    # Satellite Velocity
    sv_x = df["SvVelocityXEcefMetersPerSecond"].values
    sv_y = df["SvVelocityYEcefMetersPerSecond"].values
    sv_z = df["SvVelocityZEcefMetersPerSecond"].values

    # Theoretical Doppler (Range Rate)
    # Doppler = dot(v_sat - v_user, LOS)
    # We assume v_user is 0 for the instantaneous residual calculation.
    # Any user motion will appear in the residual, which is a useful feature.
    theo_doppler = sv_x * los_x + sv_y * los_y + sv_z * los_z

    # Measured Doppler
    meas_doppler = df["PseudorangeRateMetersPerSecond"]

    # Raw Doppler Residual = Measured - Theoretical
    # Contains Receiver Clock Drift + User Motion + Multipath
    raw_res_dop = meas_doppler - theo_doppler

    # Remove Receiver Clock Drift (Common Mode Error) per epoch
    epoch_bias_dop = raw_res_dop.groupby(df["utcTimeMillis"]).transform("median")
    res_dop = raw_res_dop - epoch_bias_dop

    # --- 4. Store Results ---
    df["los_x"] = los_x
    df["los_y"] = los_y
    df["los_z"] = los_z
    df["res_pr"] = res_pr
    df["res_dop"] = res_dop

    # Compute Weight based on Signal Strength (Cn0)
    # Simple normalization: weights roughly 0 to 5
    df["weight"] = df["Cn0DbHz"].fillna(0) / 10.0

    return df


def aggregate_forces(df):
    """
    Aggregates per-satellite residuals into per-epoch 'Force' vectors.
    Rotates these ECEF forces into the local ENU frame.

    Args:
        df (pd.DataFrame): Dataframe with residuals and LOS vectors.

    Returns:
        pd.DataFrame: Aggregated dataframe indexed by utcTimeMillis.
    """
    # Compute Weighted Force Components (ECEF)
    # Force = sum(weight * residual * LOS) / sum(weight)

    # Pseudorange Forces
    df["fx_pr"] = df["weight"] * df["res_pr"] * df["los_x"]
    df["fy_pr"] = df["weight"] * df["res_pr"] * df["los_y"]
    df["fz_pr"] = df["weight"] * df["res_pr"] * df["los_z"]

    # Doppler Forces
    df["fx_dop"] = df["weight"] * df["res_dop"] * df["los_x"]
    df["fy_dop"] = df["weight"] * df["res_dop"] * df["los_y"]
    df["fz_dop"] = df["weight"] * df["res_dop"] * df["los_z"]

    # Aggregation Dictionary
    agg_dict = {
        "fx_pr": "sum",
        "fy_pr": "sum",
        "fz_pr": "sum",
        "fx_dop": "sum",
        "fy_dop": "sum",
        "fz_dop": "sum",
        "weight": "sum",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "Cn0DbHz": ["mean", "std", "max"],
        "Svid": "count",
    }

    # Group by Epoch
    grouped = df.groupby("utcTimeMillis").agg(agg_dict)

    # Flatten MultiIndex Columns
    grouped.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in grouped.columns.values
    ]

    # Rename for clarity
    grouped.rename(
        columns={
            "WlsPositionXEcefMeters_first": "wls_x",
            "WlsPositionYEcefMeters_first": "wls_y",
            "WlsPositionZEcefMeters_first": "wls_z",
            "weight_sum": "total_weight",
        },
        inplace=True,
    )

    # Normalize Forces by Total Weight
    w = grouped["total_weight"] + 1e-6

    fx_pr_ecef = grouped["fx_pr_sum"] / w
    fy_pr_ecef = grouped["fy_pr_sum"] / w
    fz_pr_ecef = grouped["fz_pr_sum"] / w

    fx_dop_ecef = grouped["fx_dop_sum"] / w
    fy_dop_ecef = grouped["fy_dop_sum"] / w
    fz_dop_ecef = grouped["fz_dop_sum"] / w

    # --- Rotate Forces to ENU ---
    # We use the WLS position of the epoch as the reference point for rotation
    ref_x = grouped["wls_x"].values
    ref_y = grouped["wls_y"].values
    ref_z = grouped["wls_z"].values

    # Convert Reference to LLA
    ref_lat, ref_lon, ref_alt = ECEF_to_WGS84(ref_x, ref_y, ref_z)

    # Rotate PR Forces
    # We treat the force vector as a displacement from the reference point
    # ENU = ECEF_to_ENU(Ref + Force) - ECEF_to_ENU(Ref) is conceptually what we want,
    # but ECEF_to_ENU handles the rotation relative to Ref.
    # ECEF_to_ENU(x, y, z, ref_lat, ref_lon, ref_alt) computes vector from Ref to (x,y,z) in ENU.
    # So if we pass (Ref + Force) as the target point, the result is exactly the Force in ENU.

    e_pr, n_pr, u_pr = ECEF_to_ENU(
        ref_x + fx_pr_ecef,
        ref_y + fy_pr_ecef,
        ref_z + fz_pr_ecef,
        ref_lat,
        ref_lon,
        ref_alt,
    )

    # Rotate Doppler Forces
    e_dop, n_dop, u_dop = ECEF_to_ENU(
        ref_x + fx_dop_ecef,
        ref_y + fy_dop_ecef,
        ref_z + fz_dop_ecef,
        ref_lat,
        ref_lon,
        ref_alt,
    )

    # Assign Features
    grouped["Force_PR_E"] = e_pr
    grouped["Force_PR_N"] = n_pr
    grouped["Force_PR_U"] = u_pr

    grouped["Force_Dop_E"] = e_dop
    grouped["Force_Dop_N"] = n_dop
    grouped["Force_Dop_U"] = u_dop

    # Select Final Columns
    cols_to_keep = [
        "Force_PR_E",
        "Force_PR_N",
        "Force_PR_U",
        "Force_Dop_E",
        "Force_Dop_N",
        "Force_Dop_U",
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_max",
        "Svid_count",
    ]

    return grouped[cols_to_keep]


def generate_kinematic_features(gnss_df, drive_id, phone_name, load_cached_data=True):
    """
    Main pipeline function to generate kinematic features from GNSS data.
    Handles caching to disk.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data.
        drive_id (str): Identifier for the drive.
        phone_name (str): Identifier for the phone.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Aggregated feature dataframe indexed by utcTimeMillis.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"kinematic_{drive_id}_{phone_name}.parquet")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            # print(f"Loading kinematic features from {cache_file}")
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache {cache_file}: {e}. Recomputing...")

    # Compute features
    # print(f"Computing kinematic features for {drive_id} {phone_name}...")

    # 1. Apply Physics (Per Satellite)
    df_phys = apply_physics_transformations(gnss_df)

    # 2. Aggregate (Per Epoch)
    df_agg = aggregate_forces(df_phys)

    # Reset index to make utcTimeMillis a column, consistent with other dataframes
    df_agg.reset_index(inplace=True)

    # Save to cache
    try:
        df_agg.to_parquet(cache_file)
    except Exception as e:
        print(f"Failed to save cache {cache_file}: {e}")

    return df_agg
