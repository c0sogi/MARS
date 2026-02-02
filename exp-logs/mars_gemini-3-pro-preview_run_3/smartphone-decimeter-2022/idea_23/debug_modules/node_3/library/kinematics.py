import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from library.config import (
    WORKING_DIR,
    LIGHT_SPEED,
    SHAPE_WEIGHT_TDCP,
    SHAPE_WEIGHT_DOPPLER,
    RANSAC_THRESHOLD_TDCP,
    RANSAC_THRESHOLD_DOPPLER,
)


class KinematicsEngine:
    """
    Implements Stream B: Hybrid Kinematic Shape estimation.
    Computes relative displacements using TDCP (Carrier Phase) with Doppler fallback.
    """

    def __init__(self, cache_dir=os.path.join(WORKING_DIR, "kinematics_cache")):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, drive_id, phone_name):
        return os.path.join(
            self.cache_dir, f"kinematics_{drive_id}_{phone_name}.parquet"
        )

    def compute_displacements(
        self, gnss_df, drive_id, phone_name, load_cached_data=True
    ):
        """
        Computes hybrid kinematic displacements for the entire drive.

        Args:
            gnss_df (pd.DataFrame): Cleaned GNSS data.
            drive_id (str): Drive identifier.
            phone_name (str): Phone model name.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame with columns [UnixTimeMillis, dx, dy, dz, weight, method]
                          dx, dy, dz are ECEF displacements from t-1 to t.
        """
        cache_path = self._get_cache_path(drive_id, phone_name)

        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load kinematics cache: {e}")

        print(f"Computing kinematics for {drive_id}-{phone_name}...")

        # Ensure necessary columns exist
        required_cols = [
            "UnixTimeMillis",
            "Svid",
            "SignalType",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "AccumulatedDeltaRangeMeters",
            "AccumulatedDeltaRangeState",
            "PseudorangeRateMetersPerSecond",
            "PseudorangeRateUncertaintyMetersPerSecond",
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
        ]

        # Map utcTimeMillis to UnixTimeMillis if not present (standardize)
        if (
            "UnixTimeMillis" not in gnss_df.columns
            and "utcTimeMillis" in gnss_df.columns
        ):
            gnss_df["UnixTimeMillis"] = gnss_df["utcTimeMillis"]

        # Filter valid rows
        df = gnss_df.copy()

        # Sort by time
        df = df.sort_values("UnixTimeMillis")

        # Group by epoch
        epochs = df.groupby("UnixTimeMillis")
        timestamps = sorted(list(epochs.groups.keys()))

        results = []

        # Initialize previous epoch data
        prev_epoch_data = None
        prev_time = None
        prev_wls_pos = None

        for t in timestamps:
            curr_epoch_data = epochs.get_group(t)

            # We need WLS position for linearization.
            # Take the first valid WLS position in the epoch (usually constant per epoch)
            wls_x = curr_epoch_data["WlsPositionXEcefMeters"].iloc[0]
            wls_y = curr_epoch_data["WlsPositionYEcefMeters"].iloc[0]
            wls_z = curr_epoch_data["WlsPositionZEcefMeters"].iloc[0]

            if pd.isna(wls_x):
                # If WLS is missing, we can't linearize TDCP or Doppler effectively relative to user
                # Skip this epoch's displacement (break continuity)
                prev_epoch_data = None
                prev_time = None
                prev_wls_pos = None

                # Append a null record to maintain time alignment if needed, or just skip
                # We'll append a record with 0 weight
                results.append(
                    {
                        "UnixTimeMillis": t,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dz": 0.0,
                        "weight": 0.0,
                        "method": "none",
                    }
                )
                continue

            curr_wls_pos = np.array([wls_x, wls_y, wls_z])

            if prev_epoch_data is None:
                # First epoch, no displacement
                results.append(
                    {
                        "UnixTimeMillis": t,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dz": 0.0,
                        "weight": 0.0,
                        "method": "init",
                    }
                )
            else:
                dt = (t - prev_time) / 1000.0  # seconds

                # Attempt Hybrid Estimation
                disp, weight, method = self._compute_hybrid_displacement(
                    curr_epoch_data, prev_epoch_data, curr_wls_pos, prev_wls_pos, dt
                )

                results.append(
                    {
                        "UnixTimeMillis": t,
                        "dx": disp[0],
                        "dy": disp[1],
                        "dz": disp[2],
                        "weight": weight,
                        "method": method,
                    }
                )

            # Update previous state
            prev_epoch_data = curr_epoch_data
            prev_time = t
            prev_wls_pos = curr_wls_pos

        results_df = pd.DataFrame(results)

        # Save to cache
        try:
            results_df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save kinematics cache: {e}")

        return results_df

    def _compute_hybrid_displacement(self, curr_df, prev_df, curr_pos, prev_pos, dt):
        """
        Computes displacement using RANSAC on TDCP, falling back to Doppler.
        """
        # 1. Try TDCP
        # Identify common satellites (Svid + SignalType)
        # Create a unique ID for merging
        curr_df = curr_df.copy()
        prev_df = prev_df.copy()

        curr_df["sat_id"] = (
            curr_df["Svid"].astype(str) + "_" + curr_df["SignalType"].astype(str)
        )
        prev_df["sat_id"] = (
            prev_df["Svid"].astype(str) + "_" + prev_df["SignalType"].astype(str)
        )

        # Filter for valid carrier phase
        # ADR State: Bit 0 (1) = Valid. Bit 1 (2) = Reset. Bit 2 (4) = Cycle Slip.
        # We want Valid=1, Reset=0, CycleSlip=0.
        # So state & 7 should be 1. Or simply state & 1 == 1 and state & 6 == 0.

        def is_phase_valid(state):
            return (state & 1 == 1) & (state & 6 == 0)

        curr_valid = curr_df[is_phase_valid(curr_df["AccumulatedDeltaRangeState"])]
        prev_valid = prev_df[is_phase_valid(prev_df["AccumulatedDeltaRangeState"])]

        # Merge on sat_id
        merged_tdcp = pd.merge(
            curr_valid, prev_valid, on="sat_id", suffixes=("_curr", "_prev")
        )

        if len(merged_tdcp) >= 4:
            disp_tdcp = self._solve_tdcp(merged_tdcp, prev_pos)
            if disp_tdcp is not None:
                return disp_tdcp, SHAPE_WEIGHT_TDCP, "tdcp"

        # 2. Fallback to Doppler
        # Use current epoch Doppler measurements
        # Filter valid Doppler
        # Check uncertainty < threshold (e.g., 1.0 m/s)
        doppler_valid = curr_df[
            curr_df["PseudorangeRateUncertaintyMetersPerSecond"] < 1.0
        ]

        if len(doppler_valid) >= 4:
            velocity = self._solve_doppler(doppler_valid, curr_pos)
            if velocity is not None:
                disp_doppler = velocity * dt
                return disp_doppler, SHAPE_WEIGHT_DOPPLER, "doppler"

        # 3. Failure
        return np.zeros(3), 0.0, "none"

    def _solve_tdcp(self, merged_df, linearization_point):
        """
        Solves Time-Differenced Carrier Phase for displacement.
        Equation: e_t * dU - c*dt = ||S_t - U_{t-1}|| - ||S_{t-1} - U_{t-1}|| - (ADR_t - ADR_{t-1})
        """
        try:
            # Satellite Positions
            sx_curr = merged_df["SvPositionXEcefMeters_curr"].values
            sy_curr = merged_df["SvPositionYEcefMeters_curr"].values
            sz_curr = merged_df["SvPositionZEcefMeters_curr"].values

            sx_prev = merged_df["SvPositionXEcefMeters_prev"].values
            sy_prev = merged_df["SvPositionYEcefMeters_prev"].values
            sz_prev = merged_df["SvPositionZEcefMeters_prev"].values

            # User Position (Linearization point: U_{t-1})
            ux, uy, uz = linearization_point

            # Distances
            dist_curr = np.sqrt(
                (sx_curr - ux) ** 2 + (sy_curr - uy) ** 2 + (sz_curr - uz) ** 2
            )
            dist_prev = np.sqrt(
                (sx_prev - ux) ** 2 + (sy_prev - uy) ** 2 + (sz_prev - uz) ** 2
            )

            # Line of Sight vectors (at current time t)
            # e_t = (S_t - U_{t-1}) / ||S_t - U_{t-1}||
            ex = (sx_curr - ux) / dist_curr
            ey = (sy_curr - uy) / dist_curr
            ez = (sz_curr - uz) / dist_curr

            # ADR Difference
            d_adr = (
                merged_df["AccumulatedDeltaRangeMeters_curr"].values
                - merged_df["AccumulatedDeltaRangeMeters_prev"].values
            )

            # Geometry Change (Satellite motion term)
            d_geo = dist_curr - dist_prev

            # Residual (Observation for LS)
            # e * dU - c*dt = d_geo - d_adr
            y = d_geo - d_adr

            # Design Matrix
            # [ex, ey, ez, -1]
            X = np.column_stack((ex, ey, ez, -np.ones(len(y))))

            # RANSAC
            reg = RANSACRegressor(
                min_samples=4,
                residual_threshold=RANSAC_THRESHOLD_TDCP,
                random_state=42,
                max_trials=100,
            )
            reg.fit(X, y)

            # Check inlier ratio
            if np.sum(reg.inlier_mask_) < 4:
                return None

            # Coefficients: [dx, dy, dz, c_clk_drift]
            coeffs = reg.estimator_.coef_
            displacement = coeffs[:3]

            # Sanity check magnitude (e.g. max 100m/s -> 100m in 1s)
            if np.linalg.norm(displacement) > 100.0:
                return None

            return displacement

        except Exception:
            return None

    def _solve_doppler(self, df, user_pos):
        """
        Solves Doppler for velocity.
        Equation: e * v_rx - drift = v_sat * e - PRR
        """
        try:
            # Satellite Positions & Velocities
            sx = df["SvPositionXEcefMeters"].values
            sy = df["SvPositionYEcefMeters"].values
            sz = df["SvPositionZEcefMeters"].values

            vx = df["SvVelocityXEcefMetersPerSecond"].values
            vy = df["SvVelocityYEcefMetersPerSecond"].values
            vz = df["SvVelocityZEcefMetersPerSecond"].values

            # User Position
            ux, uy, uz = user_pos

            # Distance
            dist = np.sqrt((sx - ux) ** 2 + (sy - uy) ** 2 + (sz - uz) ** 2)

            # LOS vectors
            ex = (sx - ux) / dist
            ey = (sy - uy) / dist
            ez = (sz - uz) / dist

            # PRR
            prr = df["PseudorangeRateMetersPerSecond"].values

            # Satellite velocity projected on LOS
            sat_vel_proj = vx * ex + vy * ey + vz * ez

            # Observation y
            # e * v_rx - drift = sat_vel_proj - PRR
            y = sat_vel_proj - prr

            # Design Matrix
            # [ex, ey, ez, -1]
            X = np.column_stack((ex, ey, ez, -np.ones(len(y))))

            # RANSAC
            reg = RANSACRegressor(
                min_samples=4,
                residual_threshold=RANSAC_THRESHOLD_DOPPLER,
                random_state=42,
                max_trials=100,
            )
            reg.fit(X, y)

            if np.sum(reg.inlier_mask_) < 4:
                return None

            coeffs = reg.estimator_.coef_
            velocity = coeffs[:3]

            if np.linalg.norm(velocity) > 100.0:
                return None

            return velocity

        except Exception:
            return None
