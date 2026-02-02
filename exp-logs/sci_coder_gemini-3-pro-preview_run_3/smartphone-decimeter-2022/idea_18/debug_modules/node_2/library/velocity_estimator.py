import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from library.data_loader import load_drive_data
from library.gnss_physics import apply_physics_transformations

# Constants
CACHE_DIR = "./working/idea_18"
ADR_STATE_VALID = 1
ADR_STATE_RESET = 2
ADR_STATE_CYCLE_SLIP = 4
CLIGHT = 299792458.0

# RANSAC Parameters
TDCP_RESIDUAL_THRESHOLD = 0.05  # meters (phase is very precise)
DOPPLER_RESIDUAL_THRESHOLD = 1.0  # m/s
MIN_SATELLITES = 5  # Minimum satellites to solve for 4 unknowns (x, y, z, dt)


class VelocityEstimator:
    def __init__(self):
        pass

    @staticmethod
    def _solve_ransac(H, y, threshold):
        """
        Solves Hx = y using RANSAC.
        Returns: coef (x), uncertainty (sigma), inliers_count
        """
        if len(y) < MIN_SATELLITES:
            return None, np.inf, 0

        # RANSAC Regressor
        # We use a linear model. H is (n_samples, n_features), y is (n_samples,)
        # We want to find x such that H @ x ~ y
        try:
            ransac = RANSACRegressor(
                min_samples=MIN_SATELLITES,
                residual_threshold=threshold,
                max_trials=100,
                random_state=42,
            )
            ransac.fit(H, y)

            # Extract solution
            x_sol = ransac.estimator_.coef_
            # If the estimator calculates an intercept, we handle it.
            # However, our physics models usually include the clock term as a column in H.
            # So we typically want fit_intercept=False if we constructed H with a 1s column.
            # But sklearn RANSAC defaults fit_intercept=True.
            # Let's adjust H before calling this to include the clock term explicitly
            # and force intercept to 0, or let sklearn handle the intercept as the clock term.
            # In our physics formulation:
            # TDCP: e*dU - c*dt = ... -> H=[e_x, e_y, e_z], y=..., intercept = -c*dt
            # Doppler: e*v - drift = ... -> H=[e_x, e_y, e_z], y=..., intercept = -drift
            # So we can let sklearn estimate the intercept.

            velocity = x_sol  # [vx, vy, vz]
            clock_term = (
                ransac.estimator_.intercept_
            )  # This captures the common mode error

            # Estimate uncertainty based on inlier residuals
            inlier_mask = ransac.inlier_mask_
            if np.sum(inlier_mask) < MIN_SATELLITES:
                return None, np.inf, 0

            y_pred = ransac.predict(H[inlier_mask])
            residuals = y[inlier_mask] - y_pred
            sigma = np.std(residuals)

            # If sigma is too small (perfect fit), clamp it
            sigma = max(sigma, 1e-3)

            return velocity, sigma, np.sum(inlier_mask)

        except Exception as e:
            return None, np.inf, 0

    @staticmethod
    def estimate_velocity_tdcp(prev_df, curr_df):
        """
        Estimates displacement using Time-Differenced Carrier Phase (TDCP).
        Returns: (vx, vy, vz, sigma) or None
        """
        # Identify common satellites
        # Key: (Svid, ConstellationType, SignalType)
        # We create a composite key for merging

        # Prepare previous data
        prev_df = prev_df.copy()
        prev_df["key"] = (
            prev_df["Svid"].astype(str)
            + "_"
            + prev_df["ConstellationType"].astype(str)
            + "_"
            + prev_df["SignalType"].astype(str)
        )

        # Prepare current data
        curr_df = curr_df.copy()
        curr_df["key"] = (
            curr_df["Svid"].astype(str)
            + "_"
            + curr_df["ConstellationType"].astype(str)
            + "_"
            + curr_df["SignalType"].astype(str)
        )

        # Merge
        merged = pd.merge(curr_df, prev_df, on="key", suffixes=("", "_prev"))

        if merged.empty:
            return None

        # Filter for valid ADR states
        # We need valid state in both, and no reset/cycle slip in current
        # Check current state
        valid_mask = (merged["AccumulatedDeltaRangeState"] & ADR_STATE_VALID) != 0
        no_reset = (merged["AccumulatedDeltaRangeState"] & ADR_STATE_RESET) == 0
        no_slip = (merged["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP) == 0

        # Check previous state (mostly just needs to be valid)
        prev_valid = (merged["AccumulatedDeltaRangeState_prev"] & ADR_STATE_VALID) != 0

        mask = valid_mask & no_reset & no_slip & prev_valid
        subset = merged[mask]

        if len(subset) < MIN_SATELLITES:
            return None

        # Time difference
        dt = (
            subset["utcTimeMillis"].iloc[0] - subset["utcTimeMillis_prev"].iloc[0]
        ) / 1000.0
        if dt <= 0 or dt > 2.0:  # Gap too large or invalid
            return None

        # Construct Linear System
        # Equation: e_t * dU = e_t * dS - dADR + c*dt_clk
        # We solve for dU/dt (Velocity)
        # H = [e_x, e_y, e_z]
        # y = (e_t * dS - dADR) / dt
        # The intercept will capture (c*dt_clk)/dt

        # LOS vectors (current epoch)
        e_x = subset["los_x"].values
        e_y = subset["los_y"].values
        e_z = subset["los_z"].values

        # Satellite Displacement
        ds_x = (
            subset["SvPositionXEcefMeters"].values
            - subset["SvPositionXEcefMeters_prev"].values
        )
        ds_y = (
            subset["SvPositionYEcefMeters"].values
            - subset["SvPositionYEcefMeters_prev"].values
        )
        ds_z = (
            subset["SvPositionZEcefMeters"].values
            - subset["SvPositionZEcefMeters_prev"].values
        )

        # Projected Satellite Displacement
        proj_ds = e_x * ds_x + e_y * ds_y + e_z * ds_z

        # Delta ADR
        d_adr = (
            subset["AccumulatedDeltaRangeMeters"].values
            - subset["AccumulatedDeltaRangeMeters_prev"].values
        )

        # Target vector
        y = (proj_ds - d_adr) / dt

        # Design Matrix
        H = np.column_stack((e_x, e_y, e_z))

        # Solve
        velocity, sigma, inliers = VelocityEstimator._solve_ransac(
            H, y, TDCP_RESIDUAL_THRESHOLD
        )

        return velocity, sigma, inliers

    @staticmethod
    def estimate_velocity_doppler(curr_df):
        """
        Estimates velocity using Doppler measurements.
        Returns: (vx, vy, vz, sigma) or None
        """
        # Equation: -e * v_u + drift = res_dop
        # res_dop = meas - theo(v_u=0)
        # We solve for v_u
        # H = [-e_x, -e_y, -e_z]
        # y = res_dop
        # Intercept = drift

        if len(curr_df) < MIN_SATELLITES:
            return None

        # Use pre-calculated residuals from physics module
        # res_dop = meas_doppler - (v_s * e)
        # This residual equals: -v_u * e + drift
        y = curr_df["res_dop"].values

        e_x = curr_df["los_x"].values
        e_y = curr_df["los_y"].values
        e_z = curr_df["los_z"].values

        # Design Matrix
        H = np.column_stack((-e_x, -e_y, -e_z))

        # Solve
        velocity, sigma, inliers = VelocityEstimator._solve_ransac(
            H, y, DOPPLER_RESIDUAL_THRESHOLD
        )

        return velocity, sigma, inliers

    @staticmethod
    def process_drive(drive_id, phone_name, gnss_path, load_cached_data=True):
        """
        Computes velocity profile for a drive.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(
            CACHE_DIR, f"velocity_{drive_id}_{phone_name}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        # Load Data
        df = load_drive_data(
            drive_id, phone_name, gnss_path, load_cached_data=load_cached_data
        )

        # Apply Physics (get LOS, residuals)
        df_phys = apply_physics_transformations(df)

        # Group by Epoch
        groups = list(df_phys.groupby("utcTimeMillis"))

        results = []

        # Iterate
        for i in range(len(groups)):
            curr_time, curr_df = groups[i]

            # Initialize result
            res = {
                "UnixTimeMillis": curr_time,
                "v_x": np.nan,
                "v_y": np.nan,
                "v_z": np.nan,
                "uncertainty": np.nan,
                "method": 0,  # 0: Fail, 1: TDCP, 2: Doppler
            }

            # Attempt TDCP
            tdcp_success = False
            if i > 0:
                prev_time, prev_df = groups[i - 1]
                # Check time gap (max 2 seconds)
                if (curr_time - prev_time) <= 2000:
                    vel, sig, n_in = VelocityEstimator.estimate_velocity_tdcp(
                        prev_df, curr_df
                    )
                    if vel is not None:
                        res["v_x"], res["v_y"], res["v_z"] = vel
                        res["uncertainty"] = sig
                        res["method"] = 1
                        tdcp_success = True

            # Attempt Doppler if TDCP failed
            if not tdcp_success:
                vel, sig, n_in = VelocityEstimator.estimate_velocity_doppler(curr_df)
                if vel is not None:
                    res["v_x"], res["v_y"], res["v_z"] = vel
                    res["uncertainty"] = sig
                    res["method"] = 2

            results.append(res)

        result_df = pd.DataFrame(results)
        result_df.to_parquet(cache_file)
        return result_df


def compute_velocity_profile(drive_id, phone_name, gnss_path, load_cached_data=True):
    """
    Wrapper function to match the requested module interface style.
    """
    return VelocityEstimator.process_drive(
        drive_id, phone_name, gnss_path, load_cached_data
    )
