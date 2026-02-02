import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from library.config import (
    OUTPUT_DIR,
    RANSAC_THRESHOLD,
    RANSAC_MIN_SAMPLES,
    SEED,
    WGS84_A,
    WGS84_B,
)
from library.gnss_utils import ecef2enu, lla2ecef, calculate_los_vector

# Constants
CLIGHT = 299792458.0
# AccumulatedDeltaRangeState bit masks
ADR_STATE_VALID = 1
ADR_STATE_RESET = 2
ADR_STATE_CYCLE_SLIP = 4


class CarrierPhaseOdometry:
    """
    Implements Time-Differenced Carrier Phase (TDCP) odometry to estimate
    relative receiver velocity/displacement between epochs.
    """

    def __init__(self):
        pass

    def estimate_velocity_ransac(self, df_curr, df_prev, user_pos_prev_ecef):
        """
        Estimates displacement vector (dx, dy, dz) in ECEF using RANSAC on TDCP residuals.

        Args:
            df_curr: DataFrame slice for current epoch
            df_prev: DataFrame slice for previous epoch
            user_pos_prev_ecef: Tuple (x, y, z) of user position at previous epoch (WLS)

        Returns:
            dx, dy, dz (meters), weight (number of inliers)
        """
        # Merge current and previous observations on Satellite ID and Signal Type
        # This ensures we are differencing the same signal
        merged = pd.merge(
            df_curr,
            df_prev,
            on=["Svid", "SignalType"],
            suffixes=("", "_prev"),
            how="inner",
        )

        if len(merged) < RANSAC_MIN_SAMPLES:
            return 0.0, 0.0, 0.0, 0.0

        # Filter for valid Carrier Phase measurements
        # We need Valid state, No Reset, and No Cycle Slip in BOTH epochs
        def is_valid_adr(state):
            return (
                (state & ADR_STATE_VALID)
                and not (state & ADR_STATE_RESET)
                and not (state & ADR_STATE_CYCLE_SLIP)
            )

        valid_mask = merged["AccumulatedDeltaRangeState"].apply(is_valid_adr) & merged[
            "AccumulatedDeltaRangeState_prev"
        ].apply(is_valid_adr)

        valid_data = merged[valid_mask].copy()

        if len(valid_data) < RANSAC_MIN_SAMPLES:
            return 0.0, 0.0, 0.0, 0.0

        # 1. Calculate Observed Range Change (Delta ADR)
        # ADR is in meters.
        delta_adr = (
            valid_data["AccumulatedDeltaRangeMeters"]
            - valid_data["AccumulatedDeltaRangeMeters_prev"]
        )

        # 2. Calculate Geometry Change (Satellite Motion)
        # We approximate the geometry change assuming the user stayed at the previous position.
        # Any discrepancy is due to user motion + clock drift.

        # Satellite positions at transmission time
        sat_pos_curr = valid_data[
            ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
        ].values
        sat_pos_prev = valid_data[
            [
                "SvPositionXEcefMeters_prev",
                "SvPositionYEcefMeters_prev",
                "SvPositionZEcefMeters_prev",
            ]
        ].values

        u_prev_arr = np.array(user_pos_prev_ecef)  # Shape (3,)

        # Distance from Sat(t) to User(t-1)
        dist_curr = np.linalg.norm(sat_pos_curr - u_prev_arr, axis=1)
        # Distance from Sat(t-1) to User(t-1)
        dist_prev = np.linalg.norm(sat_pos_prev - u_prev_arr, axis=1)

        # Range change due to satellite motion only
        geometry_change = dist_curr - dist_prev

        # 3. Formulate Residuals
        # y = Observed_Delta - Geometry_Change
        # y = (u * dx) + c*dt + error
        y = delta_adr.values - geometry_change

        # 4. Design Matrix
        # Line of sight vectors from User(t-1) to Sat(t)
        # Using Sat(t) is standard for linearization at current epoch,
        # though average LOS is also used. Given small displacement, Sat(t) is fine.
        los_vectors = calculate_los_vector(u_prev_arr, sat_pos_curr)

        # H matrix: [LOS_x, LOS_y, LOS_z, 1]
        # The '1' accounts for the common clock drift term (c * delta_dt)
        H = np.column_stack([los_vectors, np.ones(len(y))])

        # 5. RANSAC Solver
        try:
            ransac = RANSACRegressor(
                random_state=SEED,
                min_samples=RANSAC_MIN_SAMPLES,
                residual_threshold=RANSAC_THRESHOLD,
            )
            ransac.fit(H, y)

            # Coefficients: [dx, dy, dz, c*dt]
            coeffs = ransac.estimator_.coef_

            dx, dy, dz = coeffs[0], coeffs[1], coeffs[2]

            # Weight is number of inliers
            inlier_mask = ransac.inlier_mask_
            weight = np.sum(inlier_mask)

            # Sanity check for extreme velocities (e.g. > 100 m/s is unlikely for cars)
            if np.linalg.norm([dx, dy, dz]) > 100.0:
                return 0.0, 0.0, 0.0, 0.0

            return dx, dy, dz, weight

        except Exception:
            # Fallback if RANSAC fails
            return 0.0, 0.0, 0.0, 0.0

    def process_drive(self, drive_id, phone_name, gnss_path, load_cached_data=True):
        """
        Computes the kinematic trajectory (relative displacements) for a full drive.

        Args:
            drive_id: ID of the drive
            phone_name: Name of the phone
            gnss_path: Path to device_gnss.csv
            load_cached_data: Whether to load from cache if available

        Returns:
            DataFrame with columns [UnixTimeMillis, d_E, d_N, d_U, weight]
        """
        # Setup Cache
        cache_dir = os.path.join(OUTPUT_DIR, "kin_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, f"kinematics_{drive_id}_{phone_name}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            # print(f"Loading cached kinematics for {drive_id}-{phone_name}")
            return pd.read_parquet(cache_file)

        print(f"Computing kinematics for {drive_id}-{phone_name}")

        # Load GNSS Data
        # We need specific columns. Loading all for simplicity, can optimize if memory issue.
        try:
            df_gnss = pd.read_csv(gnss_path)
        except FileNotFoundError:
            print(f"GNSS file not found: {gnss_path}")
            return pd.DataFrame()

        # Enforce numeric types for WLS columns to avoid object-type issues with np.isnan
        # Cite debug_lesson_22
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        for col in wls_cols:
            if col in df_gnss.columns:
                df_gnss[col] = pd.to_numeric(df_gnss[col], errors="coerce")

        # Ensure sorted by time
        df_gnss = df_gnss.sort_values("utcTimeMillis")

        # Group by epoch
        grouped = df_gnss.groupby("utcTimeMillis")
        timestamps = sorted(list(grouped.groups.keys()))

        results = []

        # Iterate through consecutive epochs
        for i in range(1, len(timestamps)):
            t_curr = timestamps[i]
            t_prev = timestamps[i - 1]

            # Check time gap. If > 1.5s, assume discontinuity and skip (0 displacement)
            if (t_curr - t_prev) > 1500:
                results.append(
                    {
                        "UnixTimeMillis": t_curr,
                        "d_E": 0.0,
                        "d_N": 0.0,
                        "d_U": 0.0,
                        "weight": 0.0,
                    }
                )
                continue

            df_curr = grouped.get_group(t_curr)
            df_prev = grouped.get_group(t_prev)

            # Get WLS position of previous epoch as linearization point
            # Taking the first row's WLS position (they are identical for all sats in same epoch)
            wls_pos_prev = df_prev.iloc[0][
                [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ].values

            # Check if WLS is valid (not nan)
            if np.isnan(wls_pos_prev).any():
                results.append(
                    {
                        "UnixTimeMillis": t_curr,
                        "d_E": 0.0,
                        "d_N": 0.0,
                        "d_U": 0.0,
                        "weight": 0.0,
                    }
                )
                continue

            # Estimate ECEF displacement
            dx, dy, dz, weight = self.estimate_velocity_ransac(
                df_curr, df_prev, wls_pos_prev
            )

            # Convert ECEF displacement to ENU displacement
            # We need a reference Lat/Lon/Alt. Use WLS of prev epoch.
            # We need to convert WLS ECEF to LLA first
            lat0, lon0, alt0 = ecef2lla(
                wls_pos_prev[0], wls_pos_prev[1], wls_pos_prev[2]
            )

            # Project the displacement vector (dx, dy, dz) to ENU
            # Note: ecef2enu calculates position difference.
            # Here we pass (x0+dx, y0+dy, z0+dz) as target and (x0, y0, z0) as ref
            # to get the ENU vector.
            de, dn, du = ecef2enu(
                wls_pos_prev[0] + dx,
                wls_pos_prev[1] + dy,
                wls_pos_prev[2] + dz,
                lat0,
                lon0,
                alt0,
            )

            results.append(
                {
                    "UnixTimeMillis": t_curr,
                    "d_E": de,
                    "d_N": dn,
                    "d_U": du,
                    "weight": weight,
                }
            )

        # Add the first timestamp with 0 displacement (start of track)
        if timestamps:
            results.insert(
                0,
                {
                    "UnixTimeMillis": timestamps[0],
                    "d_E": 0.0,
                    "d_N": 0.0,
                    "d_U": 0.0,
                    "weight": 0.0,
                },
            )

        result_df = pd.DataFrame(results)

        # Save to cache
        result_df.to_parquet(cache_file)

        return result_df
