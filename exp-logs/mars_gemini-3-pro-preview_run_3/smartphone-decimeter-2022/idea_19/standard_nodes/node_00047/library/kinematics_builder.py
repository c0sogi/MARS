import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from library.config import (
    WORKING_DIR,
    RANSAC_MIN_SAMPLES,
    RANSAC_RESIDUAL_THRESHOLD,
    RANSAC_MAX_TRIALS,
    SEED,
)
from library.gnss_core import calculate_los_vectors, calculate_carrier_phase_differences


def estimate_velocity_doppler(df):
    """
    Estimate receiver velocity using Weighted Least Squares on Doppler measurements.

    Equation: u * v_rx - c * dt_dot = u * v_sat - PRR

    Args:
        df: DataFrame containing device_gnss.csv data.

    Returns:
        DataFrame with columns [UnixTimeMillis, Doppler_Vel_X, Doppler_Vel_Y, Doppler_Vel_Z, Doppler_ClkDrift]
    """
    # Required columns
    req_cols = [
        "UnixTimeMillis",
        "Svid",
        "SignalType",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
    ]

    # Filter valid rows
    valid_mask = (
        df["PseudorangeRateUncertaintyMetersPerSecond"].notna()
        & (df["PseudorangeRateUncertaintyMetersPerSecond"] > 0)
        & df["PseudorangeRateMetersPerSecond"].notna()
    )

    proc_df = df.loc[valid_mask, req_cols].copy()

    if proc_df.empty:
        return pd.DataFrame(
            columns=[
                "UnixTimeMillis",
                "Doppler_Vel_X",
                "Doppler_Vel_Y",
                "Doppler_Vel_Z",
                "Doppler_ClkDrift",
            ]
        )

    # Calculate LOS vectors (u)
    rx_pos = proc_df[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sat_pos = proc_df[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values
    u, _ = calculate_los_vectors(rx_pos, sat_pos)

    # Satellite Velocity (v_sat)
    v_sat = proc_df[
        [
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
        ]
    ].values

    # Pseudorange Rate (PRR)
    prr = proc_df["PseudorangeRateMetersPerSecond"].values

    # Construct y = u * v_sat - PRR
    # (N, 3) dot (N, 3) -> (N,)
    u_dot_vsat = np.sum(u * v_sat, axis=1)
    y = u_dot_vsat - prr

    # Construct A = [u_x, u_y, u_z, -1]
    # We solve for x = [v_rx_x, v_rx_y, v_rx_z, c*dt_dot]
    # u * v_rx - c*dt_dot = y
    A = np.column_stack((u, np.full(len(y), -1.0)))

    # Weights = 1 / sigma^2
    weights = 1.0 / (proc_df["PseudorangeRateUncertaintyMetersPerSecond"].values ** 2)

    # Solve per epoch
    results = []

    # Group indices by timestamp
    # Using pandas groupby is convenient but can be slow.
    # Since data is likely sorted or we can sort it, we can iterate efficiently.
    # However, for robustness, we use groupby.
    grouped = proc_df.groupby("UnixTimeMillis")

    # To speed up, we can use the indices from the groupby object directly on the numpy arrays
    for time_ms, indices in grouped.indices.items():
        if len(indices) < 4:
            # Not enough satellites for 4 unknowns
            results.append([time_ms, np.nan, np.nan, np.nan, np.nan])
            continue

        A_epoch = A[indices]
        y_epoch = y[indices]
        w_epoch = weights[indices]

        try:
            # Weighted Least Squares
            # (A^T W A) x = A^T W y
            # Sklearn LinearRegression supports sample_weight
            reg = LinearRegression(fit_intercept=False)
            reg.fit(A_epoch, y_epoch, sample_weight=w_epoch)

            # coef_ is array of shape (4,)
            vel = reg.coef_
            results.append([time_ms, vel[0], vel[1], vel[2], vel[3]])
        except:
            results.append([time_ms, np.nan, np.nan, np.nan, np.nan])

    return pd.DataFrame(
        results,
        columns=[
            "UnixTimeMillis",
            "Doppler_Vel_X",
            "Doppler_Vel_Y",
            "Doppler_Vel_Z",
            "Doppler_ClkDrift",
        ],
    )


def estimate_displacement_tdcp(df):
    """
    Estimate receiver displacement using RANSAC on Time-Differenced Carrier Phase (TDCP).

    Equation: u * delta_r_rx - c * delta_dt = u * delta_r_sat - delta_ADR

    Args:
        df: DataFrame containing device_gnss.csv data.

    Returns:
        DataFrame with columns [UnixTimeMillis, TDCP_Disp_X, TDCP_Disp_Y, TDCP_Disp_Z, TDCP_ClkDrift, TDCP_Valid]
    """
    # 1. Calculate TDCP observables
    # This returns rows where consecutive phase measurements exist
    tdcp_df = calculate_carrier_phase_differences(df)

    if tdcp_df.empty:
        return pd.DataFrame(
            columns=[
                "UnixTimeMillis",
                "TDCP_Disp_X",
                "TDCP_Disp_Y",
                "TDCP_Disp_Z",
                "TDCP_ClkDrift",
                "TDCP_Valid",
            ]
        )

    # 2. Calculate LOS vectors (u)
    # We use the satellite position at current epoch (t) and WLS position at (t)
    # Note: Ideally we use midpoint, but at 1Hz, t is sufficient.
    # We need WLS positions joined back to tdcp_df
    # tdcp_df has [UnixTimeMillis, Svid, SignalType, ...]
    # We need to merge WLS position from original df.
    # Optimization: Create a lookup for WLS position by (UnixTimeMillis)
    # Since WLS is same for all sats in an epoch, drop duplicates
    wls_lookup = (
        df[
            [
                "UnixTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .drop_duplicates("UnixTimeMillis")
        .set_index("UnixTimeMillis")
    )

    tdcp_df = tdcp_df.join(wls_lookup, on="UnixTimeMillis", how="inner")

    rx_pos = tdcp_df[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sat_pos = tdcp_df[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values

    u, _ = calculate_los_vectors(rx_pos, sat_pos)

    # 3. Construct Equation
    # y = u * delta_S - delta_ADR
    # delta_S is SatDisp
    sat_disp = tdcp_df[["SatDisp_X", "SatDisp_Y", "SatDisp_Z"]].values
    delta_adr = tdcp_df["TDCP_Meters"].values

    u_dot_ds = np.sum(u * sat_disp, axis=1)
    y = u_dot_ds - delta_adr

    # A = [u, -1]
    # x = [delta_rx, c * delta_dt]
    A = np.column_stack((u, np.full(len(y), -1.0)))

    # 4. Solve per epoch using RANSAC
    results = []
    grouped = tdcp_df.groupby("UnixTimeMillis")

    for time_ms, indices in grouped.indices.items():
        # RANSAC requires enough samples
        if len(indices) < RANSAC_MIN_SAMPLES:
            results.append([time_ms, np.nan, np.nan, np.nan, np.nan, 0])
            continue

        A_epoch = A[indices]
        y_epoch = y[indices]

        try:
            # RANSAC Regressor
            ransac = RANSACRegressor(
                min_samples=RANSAC_MIN_SAMPLES,
                residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
                max_trials=RANSAC_MAX_TRIALS,
                random_state=SEED,
            )
            ransac.fit(A_epoch, y_epoch)

            # Check if fit was successful (estimator_ is not None)
            if ransac.estimator_ is not None:
                disp = ransac.estimator_.coef_
                # Intercept is handled by the -1 column in A, so fit_intercept should be False implicitly or handled by A
                # Sklearn RANSAC uses a base estimator (LinearRegression).
                # If we pass A with 4 cols and fit y, LinearRegression(fit_intercept=True) would add a 5th term.
                # We should set fit_intercept=False for the base estimator.

                # Re-fit with explicit base estimator to ensure fit_intercept=False
                ransac.set_params(estimator=LinearRegression(fit_intercept=False))
                ransac.fit(A_epoch, y_epoch)

                disp = ransac.estimator_.coef_

                # Check inlier ratio or count for validity
                n_inliers = np.sum(ransac.inlier_mask_)
                is_valid = 1 if n_inliers >= RANSAC_MIN_SAMPLES else 0

                if is_valid:
                    results.append([time_ms, disp[0], disp[1], disp[2], disp[3], 1])
                else:
                    results.append([time_ms, np.nan, np.nan, np.nan, np.nan, 0])
            else:
                results.append([time_ms, np.nan, np.nan, np.nan, np.nan, 0])

        except Exception:
            results.append([time_ms, np.nan, np.nan, np.nan, np.nan, 0])

    return pd.DataFrame(
        results,
        columns=[
            "UnixTimeMillis",
            "TDCP_Disp_X",
            "TDCP_Disp_Y",
            "TDCP_Disp_Z",
            "TDCP_ClkDrift",
            "TDCP_Valid",
        ],
    )


def build_kinematics(drive_id, phone_name, gnss_df, load_cached_data=True):
    """
    Build kinematic features (Velocity and Displacement) for a specific drive.
    Handles caching to disk.

    Args:
        drive_id: Drive identifier string.
        phone_name: Phone model name string.
        gnss_df: DataFrame containing raw GNSS data.
        load_cached_data: Boolean, whether to try loading from disk.

    Returns:
        DataFrame indexed by UnixTimeMillis containing kinematic features.
    """
    # Construct cache path
    cache_filename = f"kinematics_{drive_id}_{phone_name}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached kinematics for {drive_id} {phone_name}...")
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Computing kinematics for {drive_id} {phone_name}...")

    # 1. Estimate Doppler Velocity
    doppler_df = estimate_velocity_doppler(gnss_df)

    # 2. Estimate TDCP Displacement
    tdcp_df = estimate_displacement_tdcp(gnss_df)

    # 3. Merge Results
    # We want a row for every timestamp in the original GNSS data (unique epochs)
    unique_epochs = pd.DataFrame(
        gnss_df["UnixTimeMillis"].unique(), columns=["UnixTimeMillis"]
    )
    unique_epochs = unique_epochs.sort_values("UnixTimeMillis")

    merged_df = unique_epochs.merge(doppler_df, on="UnixTimeMillis", how="left")
    merged_df = merged_df.merge(tdcp_df, on="UnixTimeMillis", how="left")

    # Fill TDCP_Valid with 0 if NaN (missing epochs)
    merged_df["TDCP_Valid"] = merged_df["TDCP_Valid"].fillna(0).astype(int)

    # Save to cache
    try:
        merged_df.to_parquet(cache_path, index=False)
        print(f"Saved kinematics to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return merged_df
