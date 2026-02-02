import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from library.config import Config
from library.utils import GeoUtils


def process_trip_velocity(trip_id, trip_df, cn0_thresh):
    """
    Process a single trip to estimate velocity for each epoch using Weighted Least Squares.
    This function is defined at the module level to be picklable for joblib parallelization.

    Args:
        trip_id (str): Unique trip identifier.
        trip_df (pd.DataFrame): GNSS data for the trip.
        cn0_thresh (float): Minimum signal strength threshold.

    Returns:
        pd.DataFrame: Estimated velocity per timestamp.
    """
    # Sort by time to ensure sequential processing
    trip_df = trip_df.sort_values("utcTimeMillis")

    # Extract numpy arrays for faster access
    times = trip_df["utcTimeMillis"].values
    unique_times, indices = np.unique(times, return_index=True)

    # Prepare data arrays
    wls_pos = trip_df[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sv_pos = trip_df[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values
    sv_vel = trip_df[
        [
            "SvVelocityXEcefMetersPerSecond",
            "SvVelocityYEcefMetersPerSecond",
            "SvVelocityZEcefMetersPerSecond",
        ]
    ].values
    pr_rate = trip_df["PseudorangeRateMetersPerSecond"].values
    pr_rate_unc = trip_df["PseudorangeRateUncertaintyMetersPerSecond"].values

    results = []

    # Determine the end indices for each group
    end_indices = np.append(indices[1:], len(times))

    for i, start_idx in enumerate(indices):
        end_idx = end_indices[i]
        t = unique_times[i]

        # Get User Position (Approximate WLS position provided in logs)
        # We assume it is constant for the epoch (take the first valid one)
        user_p = wls_pos[start_idx]
        if np.isnan(user_p).any():
            results.append([t, np.nan, np.nan, np.nan, np.nan])
            continue

        # Get Satellite Data for this epoch
        s_p = sv_pos[start_idx:end_idx]
        s_v = sv_vel[start_idx:end_idx]
        p_r = pr_rate[start_idx:end_idx]
        p_r_u = pr_rate_unc[start_idx:end_idx]

        # Filter invalid measurements within the epoch
        valid_sv = (
            ~np.isnan(s_p[:, 0])
            & ~np.isnan(s_v[:, 0])
            & ~np.isnan(p_r)
            & ~np.isnan(p_r_u)
        )

        # Need at least 4 satellites for 4 unknowns (vx, vy, vz, clk_drift)
        if np.sum(valid_sv) < 4:
            results.append([t, np.nan, np.nan, np.nan, np.nan])
            continue

        s_p = s_p[valid_sv]
        s_v = s_v[valid_sv]
        p_r = p_r[valid_sv]
        p_r_u = p_r_u[valid_sv]

        # --- Formulate Weighted Least Squares ---

        # 1. Compute Line-of-Sight Unit Vectors (u)
        diff = s_p - user_p
        dist = np.linalg.norm(diff, axis=1)
        # Avoid division by zero
        dist = np.where(dist < 1e-3, 1e-3, dist)
        u = diff / dist[:, None]

        # 2. Design Matrix H
        # Equation: rho_rate = u . (v_sat - v_user) + clk_drift + noise
        # Rearranged: u . v_user - clk_drift = u . v_sat - rho_rate
        # Unknowns x = [vx, vy, vz, clk_drift]
        # LHS for x: u_x * vx + u_y * vy + u_z * vz - 1 * clk_drift
        # H rows = [u_x, u_y, u_z, -1]

        n_obs = len(p_r)
        H = np.column_stack((u, -np.ones((n_obs, 1))))

        # 3. Observation Vector y
        # y = u . v_sat - rho_rate
        u_dot_vsat = np.sum(u * s_v, axis=1)
        y = u_dot_vsat - p_r

        # 4. Weight Matrix W
        # W = diag(1 / sigma^2)
        # Add small epsilon to variance to avoid div by zero
        weights = 1.0 / (p_r_u**2 + 1e-6)

        # 5. Solve (H^T W H) x = H^T W y
        # We use sqrt(W) to transform into standard least squares: min || sqrt(W) * (Hx - y) ||^2
        sqrt_w = np.sqrt(weights)[:, None]
        H_w = H * sqrt_w
        y_w = y * sqrt_w.flatten()

        try:
            # lstsq solves the linear system efficiently
            x, residuals, rank, s = np.linalg.lstsq(H_w, y_w, rcond=None)
            results.append([t, x[0], x[1], x[2], x[3]])
        except Exception:
            results.append([t, np.nan, np.nan, np.nan, np.nan])

    res_df = pd.DataFrame(
        results, columns=["UnixTimeMillis", "v_x", "v_y", "v_z", "clk_drift"]
    )
    res_df["tripId"] = trip_id
    return res_df


class DopplerVelocityEstimator:
    """
    Estimates receiver velocity using GNSS Doppler measurements.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def estimate_velocity(self, gnss_df, split_name, load_cached_data=True):
        """
        Main method to compute velocity for all trips in the provided GNSS dataframe.

        Args:
            gnss_df (pd.DataFrame): Raw GNSS data containing 'PseudorangeRateMetersPerSecond' etc.
            split_name (str): Name of the split ('train', 'val', 'test') for caching.
            load_cached_data (bool): If True, tries to load result from parquet cache.

        Returns:
            pd.DataFrame: DataFrame with columns:
                          [tripId, UnixTimeMillis, v_x, v_y, v_z, v_east, v_north, v_up, speed]
        """
        cache_path = os.path.join(
            self.working_dir, f"{split_name}_doppler_velocity.parquet"
        )

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached Doppler velocity for {split_name} from {cache_path}..."
            )
            return pd.read_parquet(cache_path)

        print(f"Computing Doppler velocity for {split_name} from scratch...")

        # 2. Filter Data
        # We only use signals with sufficient C/N0 and reasonable uncertainty
        mask = (gnss_df["Cn0DbHz"] >= Config.DOPPLER_CN0_THRESH) & (
            gnss_df["PseudorangeRateUncertaintyMetersPerSecond"] < 10.0
        )

        df_clean = gnss_df[mask].copy()

        # Check required columns
        req_cols = [
            "tripId",
            "utcTimeMillis",
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

        missing_cols = [c for c in req_cols if c not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for Doppler estimation: {missing_cols}")

        # 3. Parallel Processing
        unique_trips = df_clean["tripId"].unique()

        # Use joblib to process each trip in parallel
        # n_jobs=-1 uses all available cores
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_trip_velocity)(
                trip, df_clean[df_clean["tripId"] == trip], Config.DOPPLER_CN0_THRESH
            )
            for trip in unique_trips
        )

        if not results:
            print(
                "Warning: No velocity results computed (possibly empty input or aggressive filtering)."
            )
            return pd.DataFrame()

        velocity_df = pd.concat(results, ignore_index=True)

        # 4. Convert ECEF Velocity to ENU Velocity
        # We need the reference position (Latitude/Longitude) to build the rotation matrix.
        # We derive this from the WLS ECEF positions provided in the logs.

        # Get unique WLS positions per timestamp
        wls_ref = df_clean[
            [
                "tripId",
                "utcTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].drop_duplicates(subset=["tripId", "utcTimeMillis"])
        wls_ref.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

        # Merge WLS positions into velocity results
        velocity_df = pd.merge(
            velocity_df, wls_ref, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Convert WLS ECEF to LLA to get Lat/Lon for rotation
        lats, lons, _ = GeoUtils.ecef_to_lla(
            velocity_df["WlsPositionXEcefMeters"].values,
            velocity_df["WlsPositionYEcefMeters"].values,
            velocity_df["WlsPositionZEcefMeters"].values,
        )

        # Calculate Rotation Matrix components
        lat_rad = np.deg2rad(lats)
        lon_rad = np.deg2rad(lons)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        vx = velocity_df["v_x"].values
        vy = velocity_df["v_y"].values
        vz = velocity_df["v_z"].values

        # Apply Rotation R * v_ecef
        # East
        ve = -sin_lon * vx + cos_lon * vy

        # North
        vn = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz

        # Up
        vu = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz

        velocity_df["v_east"] = ve
        velocity_df["v_north"] = vn
        velocity_df["v_up"] = vu

        # Calculate horizontal speed magnitude
        velocity_df["speed"] = np.sqrt(ve**2 + vn**2)

        # Clean up intermediate columns
        velocity_df.drop(
            columns=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ],
            inplace=True,
        )

        # 5. Save to Cache
        print(f"Saving Doppler velocity results to {cache_path}...")
        velocity_df.to_parquet(cache_path, index=False)

        return velocity_df
