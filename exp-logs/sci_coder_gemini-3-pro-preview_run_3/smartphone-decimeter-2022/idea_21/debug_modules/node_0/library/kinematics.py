import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression
from library.config import (
    WORKING_DIR,
    LIGHT_SPEED,
    RANSAC_THRESHOLD_METERS,
    RANSAC_MIN_SAMPLES,
    WEIGHT_TDCP,
    WEIGHT_DOPPLER,
)

# Constants for ADR State (Bitmask)
ADR_STATE_VALID = 1 << 0
ADR_STATE_RESET = 1 << 1
ADR_STATE_CYCLE_SLIP = 1 << 2


def get_los_vector(rx_pos, sat_pos):
    """
    Calculate Line-Of-Sight vector from Receiver to Satellite.
    """
    diff = sat_pos - rx_pos
    dist = np.linalg.norm(diff, axis=1)
    # Avoid division by zero
    dist = np.where(dist == 0, 1e-9, dist)
    return diff / dist[:, np.newaxis]


def solve_ransac(H, y):
    """
    Solve linear system y = Hx using RANSAC.
    Returns estimated x and boolean success flag.
    """
    if len(y) < RANSAC_MIN_SAMPLES:
        return np.zeros(H.shape[1]), False

    # Initialize RANSAC with LinearRegression
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=RANSAC_MIN_SAMPLES,
        residual_threshold=RANSAC_THRESHOLD_METERS,
        random_state=42,
    )

    try:
        ransac.fit(H, y)
        # Check if we have enough inliers
        if np.sum(ransac.inlier_mask_) < RANSAC_MIN_SAMPLES:
            return np.zeros(H.shape[1]), False
        return ransac.estimator_.coef_.flatten(), True
    except Exception:
        return np.zeros(H.shape[1]), False


def estimate_velocity_tdcp(df_prev, df_curr, dt):
    """
    Estimate displacement using Time-Differenced Carrier Phase (TDCP).

    Equation:
    u * d_rx - c * dt_clk = u * d_sat - d_Phi

    Where:
    d_rx: Receiver displacement (unknown)
    d_sat: Satellite displacement (known)
    d_Phi: Change in ADR (measured)
    u: Line of Sight vector
    """
    # Merge on Svid and SignalType to find common satellites
    # Suffix _p for previous, _c for current
    common = pd.merge(
        df_prev,
        df_curr,
        on=["Svid", "SignalType", "ConstellationType"],
        suffixes=("_p", "_c"),
    )

    if common.empty:
        return np.zeros(3), False

    # Filter valid ADR
    # Check Valid bit set, and Reset/CycleSlip bits NOT set
    valid_p = (common["AccumulatedDeltaRangeState_p"] & ADR_STATE_VALID) != 0
    valid_c = (common["AccumulatedDeltaRangeState_c"] & ADR_STATE_VALID) != 0
    no_slip_c = (
        common["AccumulatedDeltaRangeState_c"]
        & (ADR_STATE_RESET | ADR_STATE_CYCLE_SLIP)
    ) == 0

    # Note: We strictly filter current state. Previous state cycle slip doesn't invalidate the *previous* measurement value itself,
    # but a reset in between definitely invalidates the delta.
    # Simpler robust check: Both must be valid, current must not have slipped/reset relative to previous.
    mask = valid_p & valid_c & no_slip_c

    subset = common[mask].copy()

    if len(subset) < RANSAC_MIN_SAMPLES:
        return np.zeros(3), False

    # Prepare Data
    # Positions
    rx_pos_p = subset[
        [
            "WlsPositionXEcefMeters_p",
            "WlsPositionYEcefMeters_p",
            "WlsPositionZEcefMeters_p",
        ]
    ].values
    sat_pos_p = subset[
        [
            "SvPositionXEcefMeters_p",
            "SvPositionYEcefMeters_p",
            "SvPositionZEcefMeters_p",
        ]
    ].values
    sat_pos_c = subset[
        [
            "SvPositionXEcefMeters_c",
            "SvPositionYEcefMeters_c",
            "SvPositionZEcefMeters_c",
        ]
    ].values

    # Use previous LOS as approximation for the interval (standard approximation)
    u = get_los_vector(rx_pos_p, sat_pos_p)

    # Measurements
    adr_p = subset["AccumulatedDeltaRangeMeters_p"].values
    adr_c = subset["AccumulatedDeltaRangeMeters_c"].values
    d_phi = adr_c - adr_p

    # Satellite displacement
    d_sat = sat_pos_c - sat_pos_p

    # Construct Linear System: y = Hx
    # Target y = u . d_sat - d_Phi
    # H = [u_x, u_y, u_z, 1]
    # x = [dr_x, dr_y, dr_z, -c*dt_clk]

    # Dot product u . d_sat
    u_dot_d_sat = np.sum(u * d_sat, axis=1)

    y = u_dot_d_sat - d_phi

    # Design Matrix H
    # We add a column of 1s for the clock term
    H = np.column_stack((u, np.ones(len(u))))

    # Solve
    x_sol, success = solve_ransac(H, y)

    if success:
        return x_sol[:3], True  # Return displacement vector
    else:
        return np.zeros(3), False


def estimate_velocity_doppler(df_curr, dt):
    """
    Estimate displacement using Doppler (Pseudorange Rate).

    Equation:
    PRR = u . (v_rx - v_sat) + clk_drift
    PRR + u . v_sat = u . v_rx + clk_drift

    Where:
    v_rx: Receiver velocity (unknown)
    v_sat: Satellite velocity (known)
    PRR: Pseudorange Rate (measured)
    """
    # Filter valid data if necessary (usually PRR is robust, but check for NaNs)
    subset = df_curr.dropna(
        subset=[
            "PseudorangeRateMetersPerSecond",
            "SvVelocityXEcefMetersPerSecond",
            "SvPositionXEcefMeters",
        ]
    )

    if len(subset) < RANSAC_MIN_SAMPLES:
        return np.zeros(3), False

    rx_pos = subset[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sat_pos = subset[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values
    sat_vel = subset[
        [
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
        ]
    ].values
    prr = subset["PseudorangeRateMetersPerSecond"].values

    u = get_los_vector(rx_pos, sat_pos)

    # Target y = PRR + u . v_sat
    # Note: Check sign convention.
    # Standard: PRR = -u.(v_sat - v_rx) + drift = u.v_rx - u.v_sat + drift
    # => PRR + u.v_sat = u.v_rx + drift
    # This aligns with standard GNSS physics where closing speed (v_sat towards rx) reduces range.

    u_dot_v_sat = np.sum(u * sat_vel, axis=1)
    y = prr + u_dot_v_sat

    # Design Matrix H = [u_x, u_y, u_z, 1]
    H = np.column_stack((u, np.ones(len(u))))

    # Solve for velocity
    x_sol, success = solve_ransac(H, y)

    if success:
        v_rx = x_sol[:3]
        return v_rx * dt, True  # Return displacement
    else:
        return np.zeros(3), False


def compute_trajectory_deltas(drive_id, phone_name, gnss_df, load_cached_data=True):
    """
    Compute relative trajectory displacements for a drive.

    Returns:
        pd.DataFrame: Columns [UnixTimeMillis, dx, dy, dz, weight]
    """
    # Cache setup
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_file = f"kinematics_{drive_id}_{phone_name}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Recompute if load fails

    print(f"Computing kinematics for {drive_id} {phone_name}...")

    # Ensure sorted unique timestamps
    timestamps = np.sort(gnss_df["UnixTimeMillis"].unique())

    results = []

    # Group data by timestamp for fast access
    grouped = gnss_df.groupby("UnixTimeMillis")

    # Initialize previous state
    prev_time = timestamps[0]
    try:
        prev_df = grouped.get_group(prev_time)
    except KeyError:
        prev_df = pd.DataFrame()  # Should not happen based on unique()

    # First point has no delta
    results.append(
        {
            "UnixTimeMillis": prev_time,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "weight": 0.0,  # Zero weight for anchor point delta
        }
    )

    for i in range(1, len(timestamps)):
        curr_time = timestamps[i]
        try:
            curr_df = grouped.get_group(curr_time)
        except KeyError:
            curr_df = pd.DataFrame()

        dt = (curr_time - prev_time) / 1000.0  # Convert ms to seconds

        # Skip large gaps (e.g. > 5 seconds) to avoid bad linearization
        if dt <= 0 or dt > 5.0:
            results.append(
                {
                    "UnixTimeMillis": curr_time,
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": 0.0,
                    "weight": 1e-6,  # Very low confidence
                }
            )
            prev_time = curr_time
            prev_df = curr_df
            continue

        # 1. Try TDCP
        disp, success = estimate_velocity_tdcp(prev_df, curr_df, dt)

        if success:
            weight = WEIGHT_TDCP
        else:
            # 2. Fallback to Doppler
            disp, success = estimate_velocity_doppler(curr_df, dt)
            if success:
                weight = WEIGHT_DOPPLER
            else:
                # 3. Fail
                disp = np.zeros(3)
                weight = 1e-6

        results.append(
            {
                "UnixTimeMillis": curr_time,
                "dx": disp[0],
                "dy": disp[1],
                "dz": disp[2],
                "weight": weight,
            }
        )

        prev_time = curr_time
        prev_df = curr_df

    result_df = pd.DataFrame(results)

    # Save cache
    result_df.to_parquet(cache_path, index=False)

    return result_df
