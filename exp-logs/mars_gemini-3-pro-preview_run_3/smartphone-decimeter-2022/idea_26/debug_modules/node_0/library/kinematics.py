import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
from library.config import WORKING_DIR, GRAPH_PARAMS, CLLIGHT, SEED

# Constants for ADR State Bitmask
ADR_STATE_VALID = 1 << 0
ADR_STATE_RESET = 1 << 1
ADR_STATE_CYCLE_SLIP = 1 << 2


def get_line_of_sight_vectors(rx_pos, sat_pos):
    """
    Compute unit line-of-sight vectors from receiver to satellites.

    Args:
        rx_pos: (3,) array, receiver ECEF position
        sat_pos: (N, 3) array, satellite ECEF positions

    Returns:
        u: (N, 3) array of unit vectors
    """
    diff = sat_pos - rx_pos
    dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)
    return diff / dist


def solve_ransac(A, y, threshold=0.5, min_samples=4):
    """
    Solve linear system Ax = y using RANSAC.

    Returns:
        x: Solution vector (first 3 elements are velocity/displacement)
        inliers: Number of inliers
        success: Boolean indicating if solution is valid
    """
    if len(y) < min_samples:
        return np.zeros(4), 0, False

    # RANSAC Regressor
    # We use a linear model: y = A * x
    # sklearn RANSAC expects X (features) and y (targets)
    try:
        ransac = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=threshold,
            max_trials=100,
            random_state=SEED,
        )
        ransac.fit(A, y)

        # Check if we have enough inliers
        inlier_mask = ransac.inlier_mask_
        n_inliers = np.sum(inlier_mask)

        if n_inliers < min_samples:
            return np.zeros(4), n_inliers, False

        # Refit on inliers using Least Squares for better precision
        x_sol, _, _, _ = np.linalg.lstsq(A[inlier_mask], y[inlier_mask], rcond=None)

        return x_sol, n_inliers, True

    except Exception:
        return np.zeros(4), 0, False


def process_trip_kinematics(trip_df):
    """
    Process a single trip to compute kinematic vectors.
    """
    # Sort by time
    trip_df = trip_df.sort_values("utcTimeMillis").reset_index(drop=True)

    # Group by epoch
    epochs = trip_df.groupby("utcTimeMillis")
    unique_times = sorted(list(epochs.groups.keys()))

    results = []

    # Pre-fetch groups to avoid repeated indexing
    epoch_data_map = {t: trip_df.loc[indices] for t, indices in epochs.groups.items()}

    prev_time = None
    prev_data = None

    for curr_time in unique_times:
        curr_data = epoch_data_map[curr_time]

        # Initialize result row
        # kin_x, kin_y, kin_z: Displacement vector (m) from t-1 to t
        # weight: Confidence weight
        # algo: 2=TDCP, 1=Doppler, 0=None
        row = {
            "UnixTimeMillis": curr_time,
            "kin_x": 0.0,
            "kin_y": 0.0,
            "kin_z": 0.0,
            "kin_weight": 0.0,
            "kin_algo": 0,
        }

        # We need a previous epoch for TDCP
        # For Doppler, we only need current epoch, but we store it as a 'step' from prev
        # If prev_time is missing (start of trip), we can't define a step, so we skip or put 0.

        if prev_time is not None:
            dt = (curr_time - prev_time) / 1000.0  # seconds

            # Only process if continuity is reasonable (e.g., 1 second gap)
            if 0.9 < dt < 1.1:

                # --- 1. Attempt TDCP (Time-Differenced Carrier Phase) ---
                # Find common satellites
                # Merge on Svid and SignalType (assuming unique per epoch/sat)
                # We need to handle duplicates if multiple signals per sat exist.
                # Ideally we pick the best one, but inner join handles intersection.

                # Filter for valid Carrier Phase in both epochs
                # Valid: Bit 0 set, Bit 1 (Reset) unset, Bit 2 (Cycle Slip) unset
                # Mask: 1 | 2 | 4 = 7. Target: 1.

                curr_valid = curr_data[
                    (curr_data["AccumulatedDeltaRangeState"] & 7) == ADR_STATE_VALID
                ]
                prev_valid = prev_data[
                    (prev_data["AccumulatedDeltaRangeState"] & 7) == ADR_STATE_VALID
                ]

                # Merge
                common = pd.merge(
                    curr_valid,
                    prev_valid,
                    on=["Svid", "SignalType"],
                    suffixes=("", "_prev"),
                )

                tdcp_success = False

                if len(common) >= GRAPH_PARAMS["min_inliers"]:
                    # Prepare System
                    # Receiver position (WLS) for unit vectors
                    rx_x = common["WlsPositionXEcefMeters"].values
                    rx_y = common["WlsPositionYEcefMeters"].values
                    rx_z = common["WlsPositionZEcefMeters"].values
                    rx_pos = np.column_stack((rx_x, rx_y, rx_z))

                    # Satellite Positions
                    sat_x = common["SvPositionXEcefMeters"].values
                    sat_y = common["SvPositionYEcefMeters"].values
                    sat_z = common["SvPositionZEcefMeters"].values
                    sat_pos = np.column_stack((sat_x, sat_y, sat_z))

                    # Previous Sat Positions
                    sat_x_prev = common["SvPositionXEcefMeters_prev"].values
                    sat_y_prev = common["SvPositionYEcefMeters_prev"].values
                    sat_z_prev = common["SvPositionZEcefMeters_prev"].values
                    sat_pos_prev = np.column_stack((sat_x_prev, sat_y_prev, sat_z_prev))

                    # Unit vectors (u) at current time
                    u = get_line_of_sight_vectors(rx_pos, sat_pos)

                    # Measurements
                    adr_curr = common["AccumulatedDeltaRangeMeters"].values
                    adr_prev = common["AccumulatedDeltaRangeMeters_prev"].values
                    delta_adr = adr_curr - adr_prev

                    # Satellite Clock Bias difference (c * dt_sat)
                    # SvClockBiasMeters is already in meters
                    clk_bias_curr = common["SvClockBiasMeters"].values
                    clk_bias_prev = common["SvClockBiasMeters_prev"].values
                    delta_clk_sat = clk_bias_curr - clk_bias_prev

                    # Satellite Displacement
                    delta_r_sat = sat_pos - sat_pos_prev

                    # Construct Target y
                    # Equation: u * d_rx - d_clk_rx = u * d_sat - d_ADR - d_clk_sat
                    # Note: Signs depend on ADR definition. Android ADR increases as range increases?
                    # Usually ADR = -Phase * lambda. Range = -ADR + ambiguities.
                    # Change in Range ~ -(ADR_t - ADR_t-1).
                    # Let's assume standard convention: Range change = - Delta ADR.
                    # d_rho = u * (d_sat - d_rx) + d_clk_rx - d_clk_sat
                    # -d_ADR = u * d_sat - u * d_rx + d_clk_rx - d_clk_sat
                    # u * d_rx - d_clk_rx = u * d_sat + d_ADR - d_clk_sat

                    # Dot product u * d_sat
                    u_dot_dsat = np.sum(u * delta_r_sat, axis=1)

                    # Target y
                    # Note: The sign of ADR in Android might be flipped relative to standard range.
                    # Empirical testing often shows d_Range ~ -d_ADR.
                    # Let's use: y = u_dot_dsat + delta_adr - delta_clk_sat
                    y = u_dot_dsat + delta_adr - delta_clk_sat

                    # Design Matrix A: [u_x, u_y, u_z, -1]
                    A = np.column_stack((u, -np.ones(len(u))))

                    # Solve
                    x_sol, n_inliers, success = solve_ransac(
                        A,
                        y,
                        threshold=0.05,  # TDCP is very precise, tight threshold (5cm)
                        min_samples=GRAPH_PARAMS["min_inliers"],
                    )

                    if success:
                        row["kin_x"] = x_sol[0]
                        row["kin_y"] = x_sol[1]
                        row["kin_z"] = x_sol[2]
                        # Weight scales with number of inliers and method precision
                        row["kin_weight"] = GRAPH_PARAMS[
                            "kinematic_weight_tdcp"
                        ] * np.sqrt(n_inliers)
                        row["kin_algo"] = 2
                        tdcp_success = True

                # --- 2. Attempt Doppler (Fallback) ---
                if not tdcp_success:
                    # Filter valid Doppler
                    # Use PseudorangeRateUncertaintyMetersPerSecond to filter bad measurements
                    # Threshold e.g., < 10 m/s uncertainty (quite loose, but RANSAC handles outliers)
                    curr_doppler = curr_data[
                        curr_data["PseudorangeRateUncertaintyMetersPerSecond"] < 10.0
                    ]

                    if len(curr_doppler) >= GRAPH_PARAMS["min_inliers"]:
                        # Receiver Position
                        rx_pos = curr_doppler[
                            [
                                "WlsPositionXEcefMeters",
                                "WlsPositionYEcefMeters",
                                "WlsPositionZEcefMeters",
                            ]
                        ].values

                        # Satellite Position & Velocity
                        sat_pos = curr_doppler[
                            [
                                "SvPositionXEcefMeters",
                                "SvPositionYEcefMeters",
                                "SvPositionZEcefMeters",
                            ]
                        ].values
                        sat_vel = curr_doppler[
                            [
                                "SvVelocityXEcefMetersPerSecond",
                                "SvVelocityYEcefMetersPerSecond",
                                "SvVelocityZEcefMetersPerSecond",
                            ]
                        ].values

                        # Unit vectors
                        u = get_line_of_sight_vectors(rx_pos, sat_pos)

                        # Measurements
                        prr = curr_doppler["PseudorangeRateMetersPerSecond"].values

                        # Satellite Clock Drift (m/s)
                        clk_drift_sat = curr_doppler[
                            "SvClockDriftMetersPerSecond"
                        ].values

                        # Equation: PRR = u * (v_sat - v_rx) + drift_rx - drift_sat
                        # u * v_rx - drift_rx = u * v_sat - PRR - drift_sat

                        u_dot_vsat = np.sum(u * sat_vel, axis=1)
                        y = u_dot_vsat - prr - clk_drift_sat

                        # Design Matrix A: [u_x, u_y, u_z, -1]
                        A = np.column_stack((u, -np.ones(len(u))))

                        # Solve for Velocity
                        v_sol, n_inliers, success = solve_ransac(
                            A,
                            y,
                            threshold=GRAPH_PARAMS["ransac_threshold"],  # e.g. 0.5 m/s
                            min_samples=GRAPH_PARAMS["min_inliers"],
                        )

                        if success:
                            # Convert velocity to displacement: d = v * dt
                            row["kin_x"] = v_sol[0] * dt
                            row["kin_y"] = v_sol[1] * dt
                            row["kin_z"] = v_sol[2] * dt
                            row["kin_weight"] = GRAPH_PARAMS[
                                "kinematic_weight_doppler"
                            ] * np.sqrt(n_inliers)
                            row["kin_algo"] = 1

        results.append(row)
        prev_time = curr_time
        prev_data = curr_data

    return pd.DataFrame(results)


def generate_kinematics(metadata_df, split_name, load_cached_data=True):
    """
    Main function to generate kinematic constraints for a dataset split.
    """
    cache_path = os.path.join(WORKING_DIR, f"kinematics_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading Kinematics for '{split_name}' from cache...")
        return pd.read_parquet(cache_path)

    print(f"Generating Kinematics for '{split_name}'...")

    # Load GNSS Data (using data_loader logic, but we need raw access here)
    # We will iterate by tripId from metadata
    unique_trips = metadata_df["tripId"].unique()

    all_kinematics = []

    # To optimize loading, we group metadata by gnss_path (file level)
    # But processing must be done per trip (phone level)
    # Often one file = one phone, but let's be safe.

    # Group by tripId directly
    for trip_id in tqdm(unique_trips, desc=f"Processing {split_name} trips"):
        trip_meta = metadata_df[metadata_df["tripId"] == trip_id]
        if trip_meta.empty:
            continue

        # Get paths
        gnss_rel_path = trip_meta.iloc[0]["gnss_path"]
        gnss_abs_path = os.path.join("./input", gnss_rel_path)

        if not os.path.exists(gnss_abs_path):
            continue

        # Load GNSS
        # We need specific columns
        cols = [
            "utcTimeMillis",
            "AccumulatedDeltaRangeState",
            "AccumulatedDeltaRangeMeters",
            "PseudorangeRateMetersPerSecond",
            "PseudorangeRateUncertaintyMetersPerSecond",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "SvClockBiasMeters",
            "SvClockDriftMetersPerSecond",
            "Svid",
            "SignalType",
        ]

        try:
            # Read header to be safe
            header = pd.read_csv(gnss_abs_path, nrows=0).columns.tolist()
            use_cols = [c for c in cols if c in header]

            gnss_df = pd.read_csv(gnss_abs_path, usecols=use_cols)

            # Calculate Kinematics for this trip
            kin_df = process_trip_kinematics(gnss_df)
            kin_df["tripId"] = trip_id

            all_kinematics.append(kin_df)

        except Exception as e:
            print(f"Error processing trip {trip_id}: {e}")
            continue

    if not all_kinematics:
        print("Warning: No kinematics generated.")
        return pd.DataFrame()

    final_df = pd.concat(all_kinematics, ignore_index=True)

    # Save cache
    print(f"Saving Kinematics to {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df
