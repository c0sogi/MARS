import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    LIGHT_SPEED,
    RANSAC_THRESHOLD_METERS,
    RANSAC_MIN_SAMPLES,
    ODOM_RELIABILITY_HIGH,
    ODOM_RELIABILITY_LOW,
    SEED,
)
from library.utils import process_with_cache, ecef_to_wgs84, ecef_to_enu
from library.data_loader import load_dataset


class RobustOdometryEstimator:
    def __init__(self):
        self.ransac_tdcp = RANSACRegressor(
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=RANSAC_THRESHOLD_METERS,
            random_state=SEED,
        )
        # Doppler allows slightly looser threshold as it's noisier
        self.ransac_doppler = RANSACRegressor(
            min_samples=RANSAC_MIN_SAMPLES, residual_threshold=1.0, random_state=SEED
        )

    def _compute_los_and_geometry(self, user_pos, sat_pos, sat_vel=None):
        """
        Computes Line-of-Sight unit vectors and distances.
        """
        diff = sat_pos - user_pos
        dist = np.linalg.norm(diff, axis=1)
        # Avoid division by zero
        dist = np.maximum(dist, 1e-3)

        u_vec = diff / dist[:, np.newaxis]

        geom_change = None
        sat_proj_vel = None

        # If we have previous user pos (implicit in calling context) or sat velocity
        # Here we just return current geometry
        return u_vec, dist

    def process_trip(self, df_trip):
        """
        Computes odometry for a single trip.
        """
        # Sort by time
        df_trip = df_trip.sort_values("UnixTimeMillis").reset_index(drop=True)

        # Unique timestamps
        timestamps = df_trip["UnixTimeMillis"].unique()

        results = []

        # We need to maintain state between epochs
        # Group data by timestamp for fast access
        grouped = df_trip.groupby("UnixTimeMillis")

        prev_time = None
        prev_df = None

        for t in timestamps:
            curr_df = grouped.get_group(t)

            # Initialize result for this timestamp
            # odom_x/y/z represents displacement from t-1 to t
            res = {
                "tripId": curr_df["tripId"].iloc[0],
                "UnixTimeMillis": t,
                "odom_x": 0.0,
                "odom_y": 0.0,
                "odom_z": 0.0,
                "reliability": 0.0,
            }

            if prev_time is None:
                # First epoch, no displacement
                results.append(res)
                prev_time = t
                prev_df = curr_df
                continue

            dt = (t - prev_time) / 1000.0

            # Skip if gap is too large (e.g. > 3 seconds), assume lost tracking
            if dt > 3.0 or dt <= 0:
                results.append(res)
                prev_time = t
                prev_df = curr_df
                continue

            # --- Attempt 1: Time-Differenced Carrier Phase (TDCP) ---
            # Criteria:
            # 1. Same Satellite (Svid, SignalType)
            # 2. Valid ADR State in both epochs (Bit 0 set, Bits 1 & 2 unset)
            #    State 1 = Valid.
            #    We check: (State & 1) == 1 AND (State & 6) == 0.
            #    Actually AccumulatedDeltaRangeState bit map:
            #    0: Valid, 1: Reset, 2: CycleSlip.
            #    So we want (State & 1) == 1 and (State & 2) == 0 and (State & 4) == 0.
            #    Wait, standard Android:
            #    Bit 0 (1): Valid
            #    Bit 1 (2): Reset
            #    Bit 2 (4): Cycle Slip
            #    So we want (ADR_State & 7) == 1.

            # Merge current and previous on Signal Key
            # Create unique key
            curr_df = curr_df.copy()
            prev_df = prev_df.copy()

            curr_df["key"] = curr_df["Svid"].astype(str) + "_" + curr_df["SignalType"]
            prev_df["key"] = prev_df["Svid"].astype(str) + "_" + prev_df["SignalType"]

            # Filter valid ADR
            # Note: AccumulatedDeltaRangeState is often float in pandas if NaNs exist
            curr_valid_mask = (
                curr_df["AccumulatedDeltaRangeState"].fillna(0).astype(int) & 7
            ) == 1
            prev_valid_mask = (
                prev_df["AccumulatedDeltaRangeState"].fillna(0).astype(int) & 7
            ) == 1

            curr_valid = curr_df[curr_valid_mask]
            prev_valid = prev_df[prev_valid_mask]

            merged_tdcp = pd.merge(
                curr_valid, prev_valid, on="key", suffixes=("", "_prev")
            )

            success = False

            if len(merged_tdcp) >= RANSAC_MIN_SAMPLES:
                # Prepare TDCP RANSAC
                # y = lambda * (phi_t - phi_prev) - (dist_sat_t_to_user_prev - dist_sat_prev_to_user_prev)
                # Note: AccumulatedDeltaRangeMeters is already in meters (lambda * phi)

                # User position at t-1 (Approximate with WLS)
                # We need a linearization point. Using WLS of t-1 is standard.
                rx_pos_prev = (
                    prev_df[
                        [
                            "WlsPositionXEcefMeters",
                            "WlsPositionYEcefMeters",
                            "WlsPositionZEcefMeters",
                        ]
                    ]
                    .iloc[0]
                    .values
                )

                # Sat positions
                sat_pos_t = merged_tdcp[
                    [
                        "SvPositionXEcefMeters",
                        "SvPositionYEcefMeters",
                        "SvPositionZEcefMeters",
                    ]
                ].values
                sat_pos_prev = merged_tdcp[
                    [
                        "SvPositionXEcefMeters_prev",
                        "SvPositionYEcefMeters_prev",
                        "SvPositionZEcefMeters_prev",
                    ]
                ].values

                # Distances to User(t-1)
                dist_t = np.linalg.norm(sat_pos_t - rx_pos_prev, axis=1)
                dist_prev = np.linalg.norm(sat_pos_prev - rx_pos_prev, axis=1)

                # Geometry Change term
                geo_term = dist_t - dist_prev

                # Measured Change
                adr_change = (
                    merged_tdcp["AccumulatedDeltaRangeMeters"].values
                    - merged_tdcp["AccumulatedDeltaRangeMeters_prev"].values
                )
                # Note: Sign convention of ADR in Android.
                # Usually ADR increases as distance decreases? Or standard phase?
                # Android docs: "Accumulated Delta Range ... in meters".
                # Similar to Pseudorange.
                # Delta ADR ~= Delta Range + c * Delta Clk.
                # So y = adr_change - geo_term.

                y = adr_change - geo_term

                # Design Matrix X
                # H = [-u_x, -u_y, -u_z, 1]
                # u is unit vector from User(t-1) to Sat(t)
                diff = sat_pos_t - rx_pos_prev
                u_vec = diff / np.linalg.norm(diff, axis=1)[:, np.newaxis]

                X = np.column_stack((-u_vec, np.ones(len(y))))

                try:
                    self.ransac_tdcp.fit(X, y)

                    # Check inlier count
                    n_inliers = np.sum(self.ransac_tdcp.inlier_mask_)

                    if n_inliers >= RANSAC_MIN_SAMPLES:
                        # Success
                        dx, dy, dz, dt_clk = self.ransac_tdcp.estimator_.coef_
                        # Intercept is usually 0 if we include column of 1s and fit_intercept=False
                        # But sklearn fits intercept separately by default.
                        # Let's adjust: sklearn RANSAC fits y = X*coef + intercept
                        # Our model is y = (-u)*dx + 1*dt.
                        # So coef corresponds to [dx, dy, dz, dt].
                        # However, sklearn handles intercept. Let's make X just [-u] and let intercept be dt.

                        # Re-fit with explicit structure for clarity
                        # X_simple = -u_vec
                        # model: y = X_simple @ delta_x + bias

                        self.ransac_tdcp.fit(-u_vec, y)
                        if np.sum(self.ransac_tdcp.inlier_mask_) >= RANSAC_MIN_SAMPLES:
                            delta_pos = (
                                self.ransac_tdcp.estimator_.coef_
                            )  # [dx, dy, dz]
                            # bias = self.ransac_tdcp.estimator_.intercept_ # c * dt

                            res["odom_x"] = delta_pos[0]
                            res["odom_y"] = delta_pos[1]
                            res["odom_z"] = delta_pos[2]
                            res["reliability"] = ODOM_RELIABILITY_HIGH
                            success = True
                except Exception:
                    # RANSAC failed (singular matrix, etc.)
                    pass

            # --- Attempt 2: Doppler (Fallback) ---
            if not success:
                # Merge on key (don't need valid ADR, just valid Doppler)
                # Check for valid PseudorangeRate
                # Android doesn't have a specific validity flag for Rate, usually check uncertainty?
                # We'll assume if it's present and not NaN, it's usable.

                merged_dopp = pd.merge(
                    curr_df.dropna(subset=["PseudorangeRateMetersPerSecond"]),
                    prev_df[
                        ["key"]
                    ],  # Just to ensure continuity of satellites if desired, or just use all current
                    on="key",
                )
                # Actually Doppler is instantaneous velocity, doesn't strictly require previous epoch match
                # except to ensure we are tracking valid sats. But calculating velocity only needs current snapshot.
                # Using snapshot is better.

                valid_dopp = curr_df.dropna(
                    subset=[
                        "PseudorangeRateMetersPerSecond",
                        "SvVelocityXEcefMetersPerSecond",
                    ]
                )

                if len(valid_dopp) >= RANSAC_MIN_SAMPLES:
                    # y = PR_Rate - (V_sat . u)
                    # y = - (V_rx . u) + drift
                    # y = -u . V_rx + drift

                    rx_pos = (
                        curr_df[
                            [
                                "WlsPositionXEcefMeters",
                                "WlsPositionYEcefMeters",
                                "WlsPositionZEcefMeters",
                            ]
                        ]
                        .iloc[0]
                        .values
                    )
                    sat_pos = valid_dopp[
                        [
                            "SvPositionXEcefMeters",
                            "SvPositionYEcefMeters",
                            "SvPositionZEcefMeters",
                        ]
                    ].values
                    sat_vel = valid_dopp[
                        [
                            "SvVelocityXEcefMetersPerSecond",
                            "SvVelocityYEcefMetersPerSecond",
                            "SvVelocityZEcefMetersPerSecond",
                        ]
                    ].values
                    pr_rate = valid_dopp["PseudorangeRateMetersPerSecond"].values

                    # Unit vectors
                    diff = sat_pos - rx_pos
                    dist = np.linalg.norm(diff, axis=1)
                    u_vec = diff / np.maximum(dist, 1e-3)[:, np.newaxis]

                    # Project Sat Velocity
                    sat_vel_proj = np.sum(sat_vel * u_vec, axis=1)

                    # Target
                    y = pr_rate - sat_vel_proj

                    # X matrix: -u
                    X = -u_vec

                    try:
                        self.ransac_doppler.fit(X, y)
                        if (
                            np.sum(self.ransac_doppler.inlier_mask_)
                            >= RANSAC_MIN_SAMPLES
                        ):
                            v_rx = self.ransac_doppler.estimator_.coef_

                            res["odom_x"] = v_rx[0] * dt
                            res["odom_y"] = v_rx[1] * dt
                            res["odom_z"] = v_rx[2] * dt
                            res["reliability"] = ODOM_RELIABILITY_LOW
                            success = True
                    except Exception:
                        pass

            results.append(res)
            prev_time = t
            prev_df = curr_df

        return pd.DataFrame(results)


def _process_trip_wrapper(args):
    """
    Wrapper for parallel execution.
    args: (df_trip, estimator)
    """
    df_trip, estimator = args
    return estimator.process_trip(df_trip)


def _compute_odometry_dataset(split, max_drives=None):
    """
    Computes odometry for the entire dataset split.
    """
    # Load raw data
    print(f"Loading raw dataset for split: {split}...")
    df_raw = load_dataset(split, load_cached_data=True, max_drives=max_drives)

    if df_raw.empty:
        return pd.DataFrame()

    # Group by trip
    trips = list(df_raw.groupby("tripId"))
    print(f"Processing {len(trips)} trips for odometry...")

    estimator = RobustOdometryEstimator()

    # Prepare args
    # Note: Passing the estimator object might be slow if it's large, but here it's light.
    # However, RANSAC is not stateless, but we re-fit every time.
    # To be safe with threads/processes, we instantiate inside or use a clean one.
    # Since we use ProcessPool, we can just pass the class and instantiate inside,
    # or pass the instance (it will be pickled).

    # Let's run sequentially if debugging, parallel otherwise
    # Parallel is tricky with pandas groupby objects and pickling.
    # Let's try simple loop first, if slow, optimize.
    # 200k epochs ~ 1 hour in python loop might be tight.
    # Let's use ProcessPoolExecutor.

    results = []

    # We need to extract the groups into a list of DataFrames to pass to map
    trip_dfs = [group for _, group in trips]

    # Define a standalone function for the pool to avoid pickling issues with 'self' if possible
    # But we need the estimator configuration.

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map returns an iterator
        # We create a new estimator in each process implicitly by pickling
        futures = executor.map(
            _process_trip_wrapper, [(df, estimator) for df in trip_dfs]
        )

        for res_df in tqdm(futures, total=len(trip_dfs), desc="Computing Odometry"):
            results.append(res_df)

    final_df = pd.concat(results, ignore_index=True)
    return final_df


def extract_odometry(split, load_cached_data=True, max_drives=None):
    """
    Main entry point for odometry extraction.
    """
    suffix = f"_{max_drives}" if max_drives else ""
    cache_name = f"odometry_{split}{suffix}.parquet"

    return process_with_cache(
        filename=cache_name,
        processing_func=_compute_odometry_dataset,
        load_cached_data=load_cached_data,
        split=split,
        max_drives=max_drives,
    )
