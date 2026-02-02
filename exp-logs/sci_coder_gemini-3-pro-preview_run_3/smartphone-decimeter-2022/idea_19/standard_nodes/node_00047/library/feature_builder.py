import os
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    L1_BAND_MIN,
    L1_BAND_MAX,
    L5_BAND_MIN,
    L5_BAND_MAX,
    WGS84_A,
    WGS84_B,
    SEED,
)
from library.utils import wgs84_to_ecef, ecef_to_enu, ecef_to_wgs84
from library.gnss_core import calculate_pseudorange_residuals, calculate_los_vectors
from library.kinematics_builder import build_kinematics

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def rotate_vector_ecef_to_enu(vec_x, vec_y, vec_z, ref_lat, ref_lon):
    """
    Rotate a vector from ECEF frame to local ENU frame.
    """
    phi = np.deg2rad(ref_lat)
    lam = np.deg2rad(ref_lon)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotation matrix R
    # e = -sin_lam * x + cos_lam * y
    # n = -sin_phi * cos_lam * x - sin_phi * sin_lam * y + cos_phi * z
    # u = cos_phi * cos_lam * x + cos_phi * sin_lam * y + sin_phi * z

    e = -sin_lam * vec_x + cos_lam * vec_y
    n = -sin_phi * cos_lam * vec_x - sin_phi * sin_lam * vec_y + cos_phi * vec_z
    u = cos_phi * cos_lam * vec_x + cos_phi * sin_lam * vec_y + sin_phi * vec_z

    return e, n, u


def compute_doppler_residuals(df):
    """
    Compute Doppler residuals based on the estimated receiver velocity.
    Residual = (u * v_sat - PRR) - (u * v_rx - c*dt_dot)
    """
    # Check if we have velocity estimates
    if "Doppler_Vel_X" not in df.columns:
        return np.nan

    # LOS vectors
    rx_pos = df[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sat_pos = df[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values
    u, _ = calculate_los_vectors(rx_pos, sat_pos)

    # Sat Velocity
    v_sat = df[
        [
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
        ]
    ].values

    # Rx Velocity (Estimated)
    v_rx = df[["Doppler_Vel_X", "Doppler_Vel_Y", "Doppler_Vel_Z"]].fillna(0).values
    clk_drift = df["Doppler_ClkDrift"].fillna(0).values

    # PRR
    prr = df["PseudorangeRateMetersPerSecond"].values

    # Expected PRR = u * (v_sat - v_rx) + clk_drift
    # Note: Sign convention for clock drift depends on definition.
    # Usually PRR = -Doppler * lambda.
    # Standard equation: rho_rate = u * (v_sat - v_rx) + c * (dt_sat_dot - dt_rx_dot)
    # Here we lump clock terms into clk_drift solved by linear regression.

    expected_prr = np.sum(u * (v_sat - v_rx), axis=1) + clk_drift

    residuals = expected_prr - prr
    return residuals


# -----------------------------------------------------------------------------
# Feature Engineering Functions
# -----------------------------------------------------------------------------


def compute_split_band_projections(df):
    """
    Compute geometry-projected error forces split by frequency band (L1 vs L5)
    and measurement type (Pseudorange vs Doppler).
    Returns aggregated features per epoch.
    """
    # 1. Prepare Data
    # Identify Bands
    freq = df["CarrierFrequencyHz"]
    is_l1 = (freq >= L1_BAND_MIN) & (freq <= L1_BAND_MAX)
    is_l5 = (freq >= L5_BAND_MIN) & (freq <= L5_BAND_MAX)

    df["Band"] = "Other"
    df.loc[is_l1, "Band"] = "L1"
    df.loc[is_l5, "Band"] = "L5"

    # Calculate Weights (Inverse Variance)
    # Avoid division by zero
    pr_sigma = df["RawPseudorangeUncertaintyMeters"].fillna(100.0)
    df["Weight_Pr"] = 1.0 / (pr_sigma**2 + 1e-9)

    dop_sigma = df["PseudorangeRateUncertaintyMetersPerSecond"].fillna(10.0)
    df["Weight_Dop"] = 1.0 / (dop_sigma**2 + 1e-9)

    # Calculate Doppler Residuals
    df["DopplerResidual"] = compute_doppler_residuals(df)

    # Calculate LOS Vectors
    rx_pos = df[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sat_pos = df[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values
    u, _ = calculate_los_vectors(rx_pos, sat_pos)

    # Convert WLS ECEF to Lat/Lon for rotation
    wls_lat, wls_lon, _ = ecef_to_wgs84(rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2])

    # Rotate LOS vectors to ENU
    u_e, u_n, u_u = rotate_vector_ecef_to_enu(
        u[:, 0], u[:, 1], u[:, 2], wls_lat, wls_lon
    )

    df["LOS_E"] = u_e
    df["LOS_N"] = u_n
    df["LOS_U"] = u_u

    # 2. Aggregation per Epoch
    # We want to compute Sum(w * r * u) and Sum(w * u * u^T) for each band

    features = []

    # Group by Epoch
    grouped = df.groupby("UnixTimeMillis")

    for time_ms, group in grouped:
        epoch_feats = {"UnixTimeMillis": time_ms}

        for band in ["L1", "L5"]:
            band_mask = group["Band"] == band
            if not band_mask.any():
                # Fill with 0s if band not present
                for axis in ["E", "N", "U"]:
                    epoch_feats[f"{band}_PrForce_{axis}"] = 0.0
                    epoch_feats[f"{band}_DopForce_{axis}"] = 0.0
                epoch_feats[f"{band}_Hessian_Trace"] = 0.0
                continue

            sub = group[band_mask]

            # LOS Matrix (N, 3)
            U = sub[["LOS_E", "LOS_N", "LOS_U"]].values

            # -- Pseudorange Force --
            W_pr = sub["Weight_Pr"].values
            R_pr = sub["PseudorangeResidualMeters"].fillna(0).values

            # Force = Sum(w * r * u)
            # (N,) * (N,) * (N, 3) -> sum -> (3,)
            F_pr = np.sum((W_pr * R_pr)[:, np.newaxis] * U, axis=0)

            epoch_feats[f"{band}_PrForce_E"] = F_pr[0]
            epoch_feats[f"{band}_PrForce_N"] = F_pr[1]
            epoch_feats[f"{band}_PrForce_U"] = F_pr[2]

            # -- Doppler Force --
            W_dop = sub["Weight_Dop"].values
            R_dop = sub["DopplerResidual"].fillna(0).values

            F_dop = np.sum((W_dop * R_dop)[:, np.newaxis] * U, axis=0)

            epoch_feats[f"{band}_DopForce_E"] = F_dop[0]
            epoch_feats[f"{band}_DopForce_N"] = F_dop[1]
            epoch_feats[f"{band}_DopForce_U"] = F_dop[2]

            # -- Geometry (Hessian Approximation) --
            # H = A^T W A approx Sum(w * u * u^T)
            # We just take the trace or diagonal sums as features for simplicity
            # (N, 1) * (N, 3) * (N, 3) -> (N, 3) -> sum -> (3,)
            # Actually, let's just sum the weighted squared components
            H_diag = np.sum(W_pr[:, np.newaxis] * (U**2), axis=0)
            epoch_feats[f"{band}_Hessian_Trace"] = np.sum(H_diag)

        features.append(epoch_feats)

    return pd.DataFrame(features)


def aggregate_signal_context(df):
    """
    Compute global signal context features per epoch.
    """
    agg_funcs = {
        "Cn0DbHz": ["mean", "std", "max"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
        "Svid": ["count"],
    }

    # Overall Aggregation
    agg = df.groupby("UnixTimeMillis").agg(agg_funcs)
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    # Band-specific counts
    l1_mask = (df["CarrierFrequencyHz"] >= L1_BAND_MIN) & (
        df["CarrierFrequencyHz"] <= L1_BAND_MAX
    )
    l5_mask = (df["CarrierFrequencyHz"] >= L5_BAND_MIN) & (
        df["CarrierFrequencyHz"] <= L5_BAND_MAX
    )

    l1_counts = (
        df[l1_mask].groupby("UnixTimeMillis").size().reset_index(name="L1_Count")
    )
    l5_counts = (
        df[l5_mask].groupby("UnixTimeMillis").size().reset_index(name="L5_Count")
    )

    agg = agg.merge(l1_counts, on="UnixTimeMillis", how="left").fillna(0)
    agg = agg.merge(l5_counts, on="UnixTimeMillis", how="left").fillna(0)

    return agg


def build_features(drive_id, phone_name, gnss_df, load_cached_data=True):
    """
    Main pipeline to build features for a single drive.

    Args:
        drive_id: Drive ID
        phone_name: Phone Name
        gnss_df: Raw GNSS DataFrame
        load_cached_data: Whether to load from disk

    Returns:
        DataFrame with features indexed by UnixTimeMillis.
    """
    # Cache Path
    cache_filename = f"features_{drive_id}_{phone_name}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Recompute if load fails

    # 1. Preprocessing: Residuals
    # This adds 'PseudorangeResidualMeters'
    gnss_df = calculate_pseudorange_residuals(gnss_df)

    # 2. Kinematics
    # This adds 'Doppler_Vel_X/Y/Z' needed for Doppler residuals
    kinematics_df = build_kinematics(
        drive_id, phone_name, gnss_df, load_cached_data=load_cached_data
    )

    # Merge kinematics back to GNSS df for row-wise operations
    # Note: This duplicates kinematic info for every satellite row, which is fine
    gnss_df = gnss_df.merge(kinematics_df, on="UnixTimeMillis", how="left")

    # 3. Projection Features (The Core Idea)
    proj_df = compute_split_band_projections(gnss_df)

    # 4. Signal Context
    context_df = aggregate_signal_context(gnss_df)

    # 5. Merge All Features
    # Start with unique epochs from GNSS
    features_df = pd.DataFrame(
        gnss_df["UnixTimeMillis"].unique(), columns=["UnixTimeMillis"]
    )
    features_df = features_df.sort_values("UnixTimeMillis")

    features_df = features_df.merge(proj_df, on="UnixTimeMillis", how="left")
    features_df = features_df.merge(context_df, on="UnixTimeMillis", how="left")
    features_df = features_df.merge(kinematics_df, on="UnixTimeMillis", how="left")

    # 6. Add WLS Position (Baseline)
    # We take the first row per epoch since WLS is epoch-level
    wls_df = (
        gnss_df.groupby("UnixTimeMillis")[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .first()
        .reset_index()
    )
    features_df = features_df.merge(wls_df, on="UnixTimeMillis", how="left")

    # Save to cache
    features_df.to_parquet(cache_path, index=False)

    return features_df
