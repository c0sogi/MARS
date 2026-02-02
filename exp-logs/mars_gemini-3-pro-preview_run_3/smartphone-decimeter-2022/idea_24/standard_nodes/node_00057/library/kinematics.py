import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression
from library.config import Config
from library.utils import ecef_to_enu, ecef_to_llh
from library.data_io import load_drive_data


def compute_tdcp_odometry(
    drive_id: str,
    phone_name: str,
    gnss_path: str,
    imu_path: str,
    load_cached_data: bool = True,
) -> pd.DataFrame:
    """
    Computes relative odometry (delta positions) using Time-Differenced Carrier Phase (TDCP)
    with a Doppler fallback.

    Args:
        drive_id: Drive identifier.
        phone_name: Phone model name.
        gnss_path: Relative path to GNSS data.
        imu_path: Relative path to IMU data (unused here but kept for signature consistency).
        load_cached_data: Whether to load from cache if available.

    Returns:
        DataFrame with columns ['UnixTimeMillis', 'dE', 'dN', 'dU', 'weight_odom'].
        dE, dN, dU are relative displacements in meters in the local ENU frame.
    """
    # 1. Cache Check
    cache_dir = os.path.join(Config.WORKING_DIR, "kin_cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_drive = drive_id.replace("/", "_").replace("\\", "_")
    safe_phone = phone_name.replace("/", "_").replace("\\", "_")
    cache_path = os.path.join(
        cache_dir, f"kinematics_{safe_drive}_{safe_phone}.parquet"
    )

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Warning: Failed to load kinematics cache: {e}. Recomputing...")

    # 2. Load Data
    data = load_drive_data(
        drive_id, phone_name, gnss_path, imu_path, gt_path=None, load_cached_data=True
    )
    df_gnss = data["gnss"]

    if df_gnss.empty:
        return pd.DataFrame(columns=["UnixTimeMillis", "dE", "dN", "dU", "weight_odom"])

    # 3. Preprocessing
    # Ensure sorted by time
    df_gnss = df_gnss.sort_values("utcTimeMillis").reset_index(drop=True)

    # Filter invalid signals roughly (detailed filtering happens in loop)
    # Cn0 threshold
    df_gnss = df_gnss[df_gnss["Cn0DbHz"] >= Config.CN0_THRESHOLD].copy()

    # Create unique signal identifier
    df_gnss["SignalID"] = df_gnss["Svid"].astype(str) + "_" + df_gnss["SignalType"]

    # Get unique timestamps
    timestamps = df_gnss["utcTimeMillis"].unique()

    # Results storage
    results = []

    # Initialize previous state
    # We need to store data from the previous epoch to compute differences
    prev_epoch_data = None
    prev_time = None
    prev_wls_pos = None  # (x, y, z)

    # RANSAC model for TDCP
    # Residual threshold is tight (0.5m) because carrier phase is very precise
    ransac_tdcp = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=5,
        residual_threshold=0.5,
        max_trials=100,
        random_state=Config.SEED,
    )

    # RANSAC model for Doppler
    # Residual threshold is looser (3.0 m/s) as Doppler is noisier
    ransac_doppler = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=5,
        residual_threshold=3.0,
        max_trials=100,
        random_state=Config.SEED,
    )

    for curr_time in timestamps:
        curr_epoch_data = df_gnss[df_gnss["utcTimeMillis"] == curr_time]

        # Get WLS position for current epoch (used for LOS vectors and ENU conversion)
        # Assuming WlsPositionXEcefMeters exists and is populated
        # Taking the first valid WLS position in the epoch
        if "WlsPositionXEcefMeters" not in curr_epoch_data.columns:
            # Fallback if WLS columns missing (should not happen in this dataset)
            results.append(
                {
                    "UnixTimeMillis": curr_time,
                    "dE": 0.0,
                    "dN": 0.0,
                    "dU": 0.0,
                    "weight_odom": 0.0,
                }
            )
            continue

        wls_x = curr_epoch_data["WlsPositionXEcefMeters"].iloc[0]
        wls_y = curr_epoch_data["WlsPositionYEcefMeters"].iloc[0]
        wls_z = curr_epoch_data["WlsPositionZEcefMeters"].iloc[0]

        if np.isnan(wls_x):
            # If WLS is NaN, we can't compute geometry. Reset chain.
            prev_epoch_data = None
            prev_time = None
            prev_wls_pos = None
            results.append(
                {
                    "UnixTimeMillis": curr_time,
                    "dE": 0.0,
                    "dN": 0.0,
                    "dU": 0.0,
                    "weight_odom": 0.0,
                }
            )
            continue

        curr_wls_pos = np.array([wls_x, wls_y, wls_z])

        if prev_epoch_data is not None:
            dt = (curr_time - prev_time) / 1000.0

            # Skip if gap is too large (e.g., > 3 seconds) or invalid
            if dt > 3.0 or dt <= 0:
                prev_epoch_data = curr_epoch_data
                prev_time = curr_time
                prev_wls_pos = curr_wls_pos
                results.append(
                    {
                        "UnixTimeMillis": curr_time,
                        "dE": 0.0,
                        "dN": 0.0,
                        "dU": 0.0,
                        "weight_odom": 0.0,
                    }
                )
                continue

            # --- Merge Data ---
            # Merge current and previous on SignalID to find common satellites
            cols = [
                "SignalID",
                "AccumulatedDeltaRangeMeters",
                "AccumulatedDeltaRangeState",
                "SvPositionXEcefMeters",
                "SvPositionYEcefMeters",
                "SvPositionZEcefMeters",
                "PseudorangeRateMetersPerSecond",
                "SvVelocityXEcefMetersPerSecond",
                "SvVelocityYEcefMetersPerSecond",
                "SvVelocityZEcefMetersPerSecond",
            ]

            # Ensure columns exist
            available_cols = [c for c in cols if c in prev_epoch_data.columns]

            merged = pd.merge(
                prev_epoch_data[available_cols],
                curr_epoch_data[available_cols],
                on="SignalID",
                suffixes=("_prev", "_curr"),
            )

            success = False
            dx_ecef = np.zeros(3)
            weight = 0.0

            # --- 1. TDCP Attempt ---
            # Filter for valid Carrier Phase in BOTH epochs
            # State Check: Bit 0 (1) = Valid, Bit 1 (2) = Reset, Bit 2 (4) = Cycle Slip
            # We want Valid=1, Reset=0, CycleSlip=0
            if "AccumulatedDeltaRangeState_prev" in merged.columns:
                valid_mask = (
                    ((merged["AccumulatedDeltaRangeState_prev"] & 1) == 1)
                    & ((merged["AccumulatedDeltaRangeState_prev"] & 6) == 0)
                    & ((merged["AccumulatedDeltaRangeState_curr"] & 1) == 1)
                    & ((merged["AccumulatedDeltaRangeState_curr"] & 6) == 0)
                )
                tdcp_subset = merged[valid_mask].copy()
            else:
                tdcp_subset = pd.DataFrame()

            if len(tdcp_subset) >= 6:  # Minimum points for RANSAC (safe margin)
                # Satellite Positions
                sv_pos_prev = tdcp_subset[
                    [
                        "SvPositionXEcefMeters_prev",
                        "SvPositionYEcefMeters_prev",
                        "SvPositionZEcefMeters_prev",
                    ]
                ].values
                sv_pos_curr = tdcp_subset[
                    [
                        "SvPositionXEcefMeters_curr",
                        "SvPositionYEcefMeters_curr",
                        "SvPositionZEcefMeters_curr",
                    ]
                ].values

                # Line of Sight Vector u (from User at t-1 to Sat at t-1)
                # Using prev_wls_pos as linearization point
                vec_u_prev = sv_pos_prev - prev_wls_pos
                dist_prev = np.linalg.norm(vec_u_prev, axis=1)
                u_vec = vec_u_prev / dist_prev[:, np.newaxis]  # (N, 3)

                # Delta ADR (Accumulated Delta Range)
                # ADR tracks range. Delta ADR ~ Delta Range.
                delta_adr = (
                    tdcp_subset["AccumulatedDeltaRangeMeters_curr"].values
                    - tdcp_subset["AccumulatedDeltaRangeMeters_prev"].values
                )

                # Satellite motion correction: u . (Sat_curr - Sat_prev)
                delta_sat_pos = sv_pos_curr - sv_pos_prev
                proj_sat_motion = np.sum(u_vec * delta_sat_pos, axis=1)

                # TDCP Equation: u . dx_rx - c*dt = u . dSat - dADR
                # y = proj_sat_motion - delta_adr
                # X = [u_x, u_y, u_z, -1]
                # Solve for [dx, dy, dz, c*dt]

                y = proj_sat_motion - delta_adr
                X = np.hstack([u_vec, -1 * np.ones((len(y), 1))])

                try:
                    ransac_tdcp.fit(X, y)
                    coeffs = ransac_tdcp.estimator_.coef_
                    dx_ecef = coeffs[:3]
                    success = True
                    weight = 1.0  # High confidence for TDCP
                except:
                    success = False

            # --- 2. Doppler Fallback ---
            if not success and len(merged) >= 6:
                # Use all common signals (Doppler usually available even if Phase is not valid)
                sv_pos_curr = merged[
                    [
                        "SvPositionXEcefMeters_curr",
                        "SvPositionYEcefMeters_curr",
                        "SvPositionZEcefMeters_curr",
                    ]
                ].values
                sv_vel_curr = merged[
                    [
                        "SvVelocityXEcefMetersPerSecond_curr",
                        "SvVelocityYEcefMetersPerSecond_curr",
                        "SvVelocityZEcefMetersPerSecond_curr",
                    ]
                ].values
                prr = merged["PseudorangeRateMetersPerSecond_curr"].values

                # LOS at current time (approximate with current WLS)
                vec_u = sv_pos_curr - curr_wls_pos
                dist = np.linalg.norm(vec_u, axis=1)
                u_vec = vec_u / dist[:, np.newaxis]

                # Satellite Doppler component: v_sat . u
                sat_doppler = np.sum(sv_vel_curr * u_vec, axis=1)

                # Doppler Equation: PRR = (v_sat - v_rx) . u + drift
                # PRR - v_sat.u = - v_rx.u + drift
                # y = PRR - sat_doppler
                # X = [-u_x, -u_y, -u_z, 1]
                # Solve for [vx, vy, vz, drift]

                y = prr - sat_doppler
                X = np.hstack([-u_vec, np.ones((len(y), 1))])

                try:
                    ransac_doppler.fit(X, y)
                    vel_coeffs = ransac_doppler.estimator_.coef_
                    velocity_ecef = vel_coeffs[:3]
                    dx_ecef = velocity_ecef * dt
                    success = True
                    weight = 0.1  # Low confidence for Doppler
                except:
                    success = False

            if success:
                # Convert ECEF delta to ENU delta relative to prev_wls_pos
                # We calculate the ENU coordinates of (prev + dx) relative to (prev)

                # Reference point (t-1)
                lat0, lon0, alt0 = ecef_to_llh(*prev_wls_pos)

                # Target point (t) in ECEF
                target_x = prev_wls_pos[0] + dx_ecef[0]
                target_y = prev_wls_pos[1] + dx_ecef[1]
                target_z = prev_wls_pos[2] + dx_ecef[2]

                # Convert target to ENU relative to ref
                de, dn, du = ecef_to_enu(target_x, target_y, target_z, lat0, lon0, alt0)

                results.append(
                    {
                        "UnixTimeMillis": curr_time,
                        "dE": de,
                        "dN": dn,
                        "dU": du,
                        "weight_odom": weight,
                    }
                )
            else:
                # Failed to estimate
                results.append(
                    {
                        "UnixTimeMillis": curr_time,
                        "dE": 0.0,
                        "dN": 0.0,
                        "dU": 0.0,
                        "weight_odom": 0.0,  # Zero weight means graph optimizer ignores this edge
                    }
                )
        else:
            # First epoch, no previous data
            results.append(
                {
                    "UnixTimeMillis": curr_time,
                    "dE": 0.0,
                    "dN": 0.0,
                    "dU": 0.0,
                    "weight_odom": 0.0,
                }
            )

        # Update state
        prev_epoch_data = curr_epoch_data
        prev_time = curr_time
        prev_wls_pos = curr_wls_pos

    # 4. Finalize
    if not results:
        return pd.DataFrame(columns=["UnixTimeMillis", "dE", "dN", "dU", "weight_odom"])

    df_results = pd.DataFrame(results)

    # Save to cache
    try:
        df_results.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save kinematics cache: {e}")

    return df_results
