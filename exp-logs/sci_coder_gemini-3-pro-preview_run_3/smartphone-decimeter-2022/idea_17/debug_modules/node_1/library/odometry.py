import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RANSACRegressor, LinearRegression
from tqdm import tqdm

# Constants
CLIGHT = 299792458.0
ADR_STATE_VALID = 1 << 0
ADR_STATE_RESET = 1 << 1
ADR_STATE_CYCLE_SLIP = 1 << 2


class RansacTdcpSolver:
    """
    Robustly estimates relative displacement vectors using TDCP and Doppler measurements
    aided by RANSAC.
    """

    def __init__(self, tdcp_threshold=0.05, doppler_threshold=1.0):
        """
        Args:
            tdcp_threshold: RANSAC residual threshold for TDCP (meters).
            doppler_threshold: RANSAC residual threshold for Doppler (m/s).
        """
        self.tdcp_threshold = tdcp_threshold
        self.doppler_threshold = doppler_threshold

    def _is_adr_valid(self, state):
        """Checks if Accumulated Delta Range state indicates a valid, continuous phase."""
        return (
            (state & ADR_STATE_VALID)
            and not (state & ADR_STATE_RESET)
            and not (state & ADR_STATE_CYCLE_SLIP)
        )

    def solve_step(self, prev_epoch, curr_epoch, dt):
        """
        Computes the displacement vector between two epochs.

        Args:
            prev_epoch (pd.DataFrame): GNSS data for time t-1.
            curr_epoch (pd.DataFrame): GNSS data for time t.
            dt (float): Time delta in seconds.

        Returns:
            tuple: (dx, dy, dz, weight, method)
                   dx, dy, dz: Displacement in ECEF meters.
                   weight: Confidence score (number of inliers).
                   method: 'tdcp' or 'doppler' or 'none'.
        """
        # User position anchor (use WLS from current epoch for geometry linearization)
        # Taking the mean WLS position of the current epoch as the linearization point
        user_x = curr_epoch["WlsPositionXEcefMeters"].iloc[0]
        user_y = curr_epoch["WlsPositionYEcefMeters"].iloc[0]
        user_z = curr_epoch["WlsPositionZEcefMeters"].iloc[0]

        if np.isnan(user_x):
            return 0.0, 0.0, 0.0, 0.0, "none"

        # --- 1. Try TDCP (Time-Differenced Carrier Phase) ---
        # Identify common satellites with valid phase in both epochs
        keys = ["Svid", "ConstellationType", "SignalType"]

        # Filter rows with valid ADR
        prev_valid = prev_epoch[
            prev_epoch["AccumulatedDeltaRangeState"].apply(self._is_adr_valid)
        ]
        curr_valid = curr_epoch[
            curr_epoch["AccumulatedDeltaRangeState"].apply(self._is_adr_valid)
        ]

        merged = pd.merge(prev_valid, curr_valid, on=keys, suffixes=("_prev", "_curr"))

        # Need at least 4 satellites for a solution (3 coords + 1 clock drift)
        # We ask for 5 to have at least some redundancy for RANSAC
        if len(merged) >= 5:
            # Satellite positions at current epoch (Transmission time)
            sx = merged["SvPositionXEcefMeters_curr"].values
            sy = merged["SvPositionYEcefMeters_curr"].values
            sz = merged["SvPositionZEcefMeters_curr"].values

            # Range from user to satellite
            ranges = np.sqrt(
                (sx - user_x) ** 2 + (sy - user_y) ** 2 + (sz - user_z) ** 2
            )

            # Line-of-sight unit vectors (u)
            ux = (sx - user_x) / ranges
            uy = (sy - user_y) / ranges
            uz = (sz - user_z) / ranges

            # Satellite displacement vector (S_k - S_{k-1})
            dsx = sx - merged["SvPositionXEcefMeters_prev"].values
            dsy = sy - merged["SvPositionYEcefMeters_prev"].values
            dsz = sz - merged["SvPositionZEcefMeters_prev"].values

            # Measured Delta Range (ADR difference)
            d_adr = (
                merged["AccumulatedDeltaRangeMeters_curr"].values
                - merged["AccumulatedDeltaRangeMeters_prev"].values
            )

            # Predicted range change due to satellite motion only: u dot dS
            # Note: This is an approximation.
            # Exact: ||S_k - U_{k-1}|| - ||S_{k-1} - U_{k-1}||
            # Linearized: u_k dot (S_k - S_{k-1})
            geom_change_sat = ux * dsx + uy * dsy + uz * dsz

            # Formulation:
            # d_adr ~= geom_change_sat - (u dot d_user) + c * d_dt
            # u dot d_user - c * d_dt = geom_change_sat - d_adr
            # Let x = [dx, dy, dz] and bias = -c * d_dt
            # Linear model: A * x + bias = y
            # Where A = [ux, uy, uz], y = geom_change_sat - d_adr

            A = np.column_stack((ux, uy, uz))
            y = geom_change_sat - d_adr

            try:
                # RANSAC for robust estimation
                ransac = RANSACRegressor(
                    estimator=LinearRegression(fit_intercept=True),
                    min_samples=4,
                    residual_threshold=self.tdcp_threshold,
                    random_state=42,
                    stop_probability=0.99,
                )
                ransac.fit(A, y)

                inliers = np.sum(ransac.inlier_mask_)
                if inliers >= 4:
                    dx, dy, dz = ransac.estimator_.coef_
                    # Weight is proportional to number of inliers
                    return dx, dy, dz, float(inliers), "tdcp"
            except Exception:
                pass  # Fallback to Doppler

        # --- 2. Try Doppler (Velocity) ---
        # Filter valid Doppler measurements
        # Use uncertainty as a basic filter
        valid_doppler = curr_epoch[
            curr_epoch["PseudorangeRateUncertaintyMetersPerSecond"] < 1.0
        ]

        if len(valid_doppler) >= 5:
            sx = valid_doppler["SvPositionXEcefMeters"].values
            sy = valid_doppler["SvPositionYEcefMeters"].values
            sz = valid_doppler["SvPositionZEcefMeters"].values

            ranges = np.sqrt(
                (sx - user_x) ** 2 + (sy - user_y) ** 2 + (sz - user_z) ** 2
            )
            ux = (sx - user_x) / ranges
            uy = (sy - user_y) / ranges
            uz = (sz - user_z) / ranges

            vsx = valid_doppler["SvVelocityXEcefMetersPerSecond"].values
            vsy = valid_doppler["SvVelocityYEcefMetersPerSecond"].values
            vsz = valid_doppler["SvVelocityZEcefMetersPerSecond"].values

            pr_rate = valid_doppler["PseudorangeRateMetersPerSecond"].values

            # Formulation:
            # pr_rate = - (v_sat - v_user) dot u + drift
            # pr_rate = - v_sat dot u + v_user dot u + drift
            # v_user dot u + drift = pr_rate + v_sat dot u

            vsat_dot_u = vsx * ux + vsy * uy + vsz * uz
            y = pr_rate + vsat_dot_u
            A = np.column_stack((ux, uy, uz))

            try:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(fit_intercept=True),
                    min_samples=4,
                    residual_threshold=self.doppler_threshold,
                    random_state=42,
                    stop_probability=0.99,
                )
                ransac.fit(A, y)

                inliers = np.sum(ransac.inlier_mask_)
                if inliers >= 4:
                    vx, vy, vz = ransac.estimator_.coef_
                    # Convert velocity to displacement
                    dx, dy, dz = vx * dt, vy * dt, vz * dt
                    # Lower weight for Doppler compared to TDCP (e.g., 0.1x)
                    return dx, dy, dz, float(inliers) * 0.1, "doppler"
            except Exception:
                pass

        return 0.0, 0.0, 0.0, 0.0, "none"

    def process_drive(self, df_gnss):
        """
        Process a single drive's GNSS data to compute odometry.
        """
        # Ensure sorted by time
        df_gnss = df_gnss.sort_values("utcTimeMillis").reset_index(drop=True)

        # Group by epoch
        grouped = df_gnss.groupby("utcTimeMillis")
        timestamps = sorted(list(grouped.groups.keys()))

        results = []

        # Initialize with 0 displacement for the first epoch
        results.append(
            {
                "UnixTimeMillis": timestamps[0],
                "dx_ecef": 0.0,
                "dy_ecef": 0.0,
                "dz_ecef": 0.0,
                "weight": 0.0,
                "method": "init",
            }
        )

        for i in range(1, len(timestamps)):
            curr_ts = timestamps[i]
            prev_ts = timestamps[i - 1]

            dt_ms = curr_ts - prev_ts
            dt_sec = dt_ms / 1000.0

            # Skip large gaps (e.g., > 3 seconds) to avoid bad integration
            if dt_sec > 3.0 or dt_sec <= 0:
                results.append(
                    {
                        "UnixTimeMillis": curr_ts,
                        "dx_ecef": 0.0,
                        "dy_ecef": 0.0,
                        "dz_ecef": 0.0,
                        "weight": 0.0,
                        "method": "gap",
                    }
                )
                continue

            prev_epoch = grouped.get_group(prev_ts)
            curr_epoch = grouped.get_group(curr_ts)

            dx, dy, dz, w, method = self.solve_step(prev_epoch, curr_epoch, dt_sec)

            results.append(
                {
                    "UnixTimeMillis": curr_ts,
                    "dx_ecef": dx,
                    "dy_ecef": dy,
                    "dz_ecef": dz,
                    "weight": w,
                    "method": method,
                }
            )

        return pd.DataFrame(results)


def run_odometry_processing(metadata_df, load_cached_data=True):
    """
    Main entry point to process all drives in the metadata.
    Handles caching of results.

    Args:
        metadata_df (pd.DataFrame): Metadata containing drive_id, phone_name, gnss_path.
        load_cached_data (bool): Whether to load from disk if available.

    Returns:
        pd.DataFrame: Concatenated odometry results for all drives.
    """
    CACHE_DIR = "./working/idea_17"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Identify unique trips
    trips = metadata_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    all_results = []

    print(f"Processing odometry for {len(trips)} trips...")

    solver = RansacTdcpSolver()

    for _, row in tqdm(trips.iterrows(), total=len(trips)):
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_rel_path = row["gnss_path"]

        trip_id = f"{drive_id}_{phone_name}"
        cache_path = os.path.join(CACHE_DIR, f"odom_{trip_id}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            df_odom = pd.read_parquet(cache_path)
        else:
            # Load GNSS data
            gnss_abs_path = os.path.join("./input", gnss_rel_path)
            if not os.path.exists(gnss_abs_path):
                print(f"Warning: GNSS file not found: {gnss_abs_path}")
                continue

            # Read necessary columns only to save memory
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
                "Svid",
                "ConstellationType",
                "SignalType",
            ]

            try:
                df_gnss = pd.read_csv(gnss_abs_path, usecols=lambda c: c in cols)

                # Process
                df_odom = solver.process_drive(df_gnss)

                # Add identifiers
                df_odom["tripId"] = (
                    f"{drive_id}-{phone_name}"  # Match competition format roughly
                )

                # Save cache
                df_odom.to_parquet(cache_path)

            except Exception as e:
                print(f"Error processing {trip_id}: {e}")
                continue

        all_results.append(df_odom)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
