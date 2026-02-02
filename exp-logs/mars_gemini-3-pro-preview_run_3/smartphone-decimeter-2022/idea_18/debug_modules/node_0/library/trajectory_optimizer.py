import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from library.coordinate_utils import WGS84_to_ECEF, ECEF_to_ENU, ENU_to_WGS84
from library.velocity_estimator import compute_velocity_profile
from library.data_loader import load_drive_data

# Constants
CACHE_DIR = "./working/idea_18"
os.makedirs(CACHE_DIR, exist_ok=True)

# Hyperparameters for Optimization
SIGMA_ANCHOR = 5.0  # Standard deviation for ML prediction anchors (meters)
SIGMA_TDCP = 0.05  # Standard deviation for TDCP velocity (m/s)
SIGMA_DOPPLER = 2.0  # Standard deviation for Doppler velocity (m/s)
HUBER_DELTA = 1.35  # Delta parameter for Huber loss on anchors


def _rotate_velocity_ecef_to_enu(v_x, v_y, v_z, ref_lat, ref_lon, ref_alt):
    """
    Rotates an ECEF velocity vector into the local ENU frame.
    Uses the linearity of the rotation: ENU(P + V) - ENU(P) = Rot(V).
    """
    # 1. Get ECEF of reference point
    ref_x, ref_y, ref_z = WGS84_to_ECEF(ref_lat, ref_lon, ref_alt)

    # 2. Project (Ref + V) to ENU
    # ECEF_to_ENU computes R * (Target - Ref)
    # If Target = Ref + V, then Target - Ref = V, so we get R * V
    ve, vn, vu = ECEF_to_ENU(
        ref_x + v_x, ref_y + v_y, ref_z + v_z, ref_lat, ref_lon, ref_alt
    )
    return ve, vn, vu


def _build_optimization_problem(params, n_epochs, anchors, odometry_edges):
    """
    Constructs the residual vector for least_squares.

    Args:
        params: Flattened state vector [e_0, n_0, ..., e_N, n_N].
        n_epochs: Number of epochs.
        anchors: List of (index, ml_e, ml_n, weight).
        odometry_edges: List of (idx_prev, idx_curr, dt, v_e, v_n, weight).

    Returns:
        residuals: 1D array of residuals.
    """
    # Reshape params to (N, 2)
    states = params.reshape((n_epochs, 2))

    residuals = []

    # 1. Anchor Residuals (Position)
    # r = (state - prediction) * weight
    # We compute separate residuals for East and North to allow component-wise loss application
    for idx, ml_e, ml_n, w in anchors:
        est_e, est_n = states[idx]
        residuals.append((est_e - ml_e) * w)
        residuals.append((est_n - ml_n) * w)

    # 2. Odometry Residuals (Velocity/Motion)
    # r = ((state_curr - state_prev) - velocity * dt) * weight
    for idx_prev, idx_curr, dt, v_e, v_n, w in odometry_edges:
        prev_e, prev_n = states[idx_prev]
        curr_e, curr_n = states[idx_curr]

        # Expected displacement
        d_e = v_e * dt
        d_n = v_n * dt

        # Residual
        res_e = (curr_e - prev_e) - d_e
        res_n = (curr_n - prev_n) - d_n

        residuals.append(res_e * w)
        residuals.append(res_n * w)

    return np.array(residuals)


def optimize_drive_trajectory(
    drive_id, phone_name, ml_preds_df, gnss_path, load_cached_data=True
):
    """
    Optimizes the trajectory for a single drive using Graph Optimization.

    Args:
        drive_id (str): Drive ID.
        phone_name (str): Phone name.
        ml_preds_df (pd.DataFrame): DataFrame with columns [UnixTimeMillis, LatitudeDegrees, LongitudeDegrees].
        gnss_path (str): Path to GNSS log file.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        pd.DataFrame: Optimized trajectory with columns [UnixTimeMillis, LatitudeDegrees, LongitudeDegrees].
    """
    cache_file = os.path.join(CACHE_DIR, f"opt_{drive_id}_{phone_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        # print(f"Loading optimized trajectory from {cache_file}")
        return pd.read_parquet(cache_file)

    # 1. Load Velocity Profile (Odometry)
    # This uses the RANSAC TDCP/Doppler estimator
    vel_df = compute_velocity_profile(
        drive_id, phone_name, gnss_path, load_cached_data=load_cached_data
    )

    # 2. Align Data
    # We need a common time index. The ML predictions define the required output timestamps.
    # Note: vel_df might have gaps or different timestamps if raw data was filtered.
    # We merge on UnixTimeMillis.

    # Ensure sorted
    ml_preds_df = ml_preds_df.sort_values("UnixTimeMillis").reset_index(drop=True)

    # 3. Define Local Coordinate System (ENU)
    # Use the first ML prediction as the reference origin to keep numbers small
    ref_lat = ml_preds_df["LatitudeDegrees"].iloc[0]
    ref_lon = ml_preds_df["LongitudeDegrees"].iloc[0]
    ref_alt = 0.0  # Altitude doesn't affect horizontal ENU projection significantly for small areas

    # 4. Prepare Optimization Inputs
    n_epochs = len(ml_preds_df)

    # Map timestamp to index
    time_to_idx = {t: i for i, t in enumerate(ml_preds_df["UnixTimeMillis"])}

    # Convert ML Lat/Lon to ENU (Anchors)
    # We assume ML preds are WGS84.
    ml_x, ml_y, ml_z = WGS84_to_ECEF(
        ml_preds_df["LatitudeDegrees"].values,
        ml_preds_df["LongitudeDegrees"].values,
        np.zeros(n_epochs),
    )
    ml_e, ml_n, _ = ECEF_to_ENU(ml_x, ml_y, ml_z, ref_lat, ref_lon, ref_alt)

    # Build Anchor List
    # Format: (index, e, n, weight)
    # Weight = 1 / sigma
    w_anchor = 1.0 / SIGMA_ANCHOR
    anchors = []
    for i in range(n_epochs):
        anchors.append((i, ml_e[i], ml_n[i], w_anchor))

    # Build Odometry Edge List
    # Format: (idx_prev, idx_curr, dt, v_e, v_n, weight)
    odometry_edges = []

    # Filter valid velocity estimates
    valid_vel = vel_df.dropna(subset=["v_x", "v_y", "v_z"])

    for _, row in valid_vel.iterrows():
        t_curr = int(row["UnixTimeMillis"])

        # We need to find the previous timestamp in our ML sequence that corresponds to this velocity step
        # Velocity at t_curr implies motion from t_prev to t_curr.
        # However, our velocity_estimator computes v at t based on (t, t-1).
        # We need to link indices in our state vector.

        if t_curr not in time_to_idx:
            continue

        idx_curr = time_to_idx[t_curr]
        if idx_curr == 0:
            continue

        idx_prev = idx_curr - 1
        t_prev = ml_preds_df["UnixTimeMillis"].iloc[idx_prev]

        dt = (t_curr - t_prev) / 1000.0

        # Sanity check on dt (e.g., if missing epochs in ML preds)
        if dt <= 0 or dt > 3.0:
            continue

        # Rotate ECEF velocity to ENU
        v_e, v_n, _ = _rotate_velocity_ecef_to_enu(
            row["v_x"], row["v_y"], row["v_z"], ref_lat, ref_lon, ref_alt
        )

        # Determine weight based on method
        method = row["method"]
        if method == 1:  # TDCP
            sigma = SIGMA_TDCP
        else:  # Doppler
            sigma = SIGMA_DOPPLER

        # Adjust weight by uncertainty reported by RANSAC if available and reasonable
        if not np.isnan(row["uncertainty"]) and row["uncertainty"] > 0:
            # Blend fixed prior with measured uncertainty
            sigma = np.sqrt(sigma**2 + row["uncertainty"] ** 2)

        w_odom = 1.0 / sigma

        odometry_edges.append((idx_prev, idx_curr, dt, v_e, v_n, w_odom))

    # 5. Run Optimization
    # Initial guess: ML predictions
    x0 = np.column_stack((ml_e, ml_n)).flatten()

    # We use 'soft_l1' loss (Huber-like) for anchors to handle outliers
    # We use 'linear' (L2) for odometry as it comes from RANSAC and should be clean
    # Since least_squares applies one loss to all residuals, we have to be clever.
    # Alternatively, we rely on the weights.
    # A common trick is to use Huber for everything, but set delta high for constraints we trust.
    # Here, we will use Huber loss with delta=1.35 (standard).
    # High weight on TDCP makes it act like a hard constraint within the linear region of Huber.

    res = least_squares(
        _build_optimization_problem,
        x0,
        args=(n_epochs, anchors, odometry_edges),
        loss="huber",
        f_scale=HUBER_DELTA,
        verbose=0,
        ftol=1e-4,
        xtol=1e-4,
        max_nfev=50,
    )

    # 6. Extract Result
    opt_states = res.x.reshape((n_epochs, 2))
    opt_e = opt_states[:, 0]
    opt_n = opt_states[:, 1]

    # 7. Convert ENU -> Lat/Lon
    opt_lat, opt_lon, _ = ENU_to_WGS84(
        opt_e, opt_n, np.zeros(n_epochs), ref_lat, ref_lon, ref_alt
    )

    # 8. Create Result DataFrame
    result_df = ml_preds_df.copy()
    result_df["LatitudeDegrees"] = opt_lat
    result_df["LongitudeDegrees"] = opt_lon

    # Save to cache
    result_df.to_parquet(cache_file)

    return result_df


def apply_trajectory_optimization(submission_df, load_cached_data=True):
    """
    Applies graph optimization to all trips in the submission dataframe.

    Args:
        submission_df (pd.DataFrame): Initial predictions (ML output).
        load_cached_data (bool): Cache flag.

    Returns:
        pd.DataFrame: Optimized submission dataframe.
    """
    from library.data_loader import load_metadata

    # Load test metadata to get file paths
    meta_df = load_metadata("test")

    # Map tripId to metadata
    trip_meta = meta_df.set_index("tripId")[
        ["drive_id", "phone_name", "gnss_path"]
    ].to_dict("index")

    optimized_dfs = []

    # Group by tripId
    for trip_id, group in submission_df.groupby("tripId"):
        if trip_id not in trip_meta:
            print(f"Warning: No metadata for {trip_id}, skipping optimization.")
            optimized_dfs.append(group)
            continue

        meta = trip_meta[trip_id]

        # Optimize
        opt_df = optimize_drive_trajectory(
            drive_id=meta["drive_id"],
            phone_name=meta["phone_name"],
            ml_preds_df=group,
            gnss_path=meta["gnss_path"],
            load_cached_data=load_cached_data,
        )

        optimized_dfs.append(opt_df)

    final_submission = pd.concat(optimized_dfs, ignore_index=True)
    return final_submission
