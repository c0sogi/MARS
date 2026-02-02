import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import WGS84_to_ECEF


class TDCPEngine:
    """
    Implements Time-Differenced Carrier Phase (TDCP) algorithms to estimate
    precise relative velocity/displacement vectors between GNSS epochs.
    """

    def __init__(self):
        self.min_sats = Config.TDCP_MIN_SATS
        self.c = Config.LIGHT_SPEED
        self.valid_mask = Config.TDCP_VALID_MASK
        self.invalid_mask = Config.TDCP_INVALID_MASK

    def _is_valid_adr(self, state):
        """
        Check if Accumulated Delta Range state is valid for TDCP.
        Must have VALID bit set, and NO Reset/CycleSlip bits.
        """
        return (state & self.valid_mask) != 0 and (state & self.invalid_mask) == 0

    def solve_step(self, df_prev, df_curr, xyz_prev):
        """
        Compute displacement between two epochs using WLS on TDCP residuals.

        Args:
            df_prev (pd.DataFrame): GNSS data for previous epoch.
            df_curr (pd.DataFrame): GNSS data for current epoch.
            xyz_prev (tuple): (x, y, z) ECEF coordinates of receiver at prev epoch.

        Returns:
            tuple: (dx, dy, dz, dt_drift, num_sats) or (NaN, NaN, NaN, NaN, 0)
        """
        # Merge on Svid and SignalType to find common observations
        # Suffix _p for prev, _c for curr
        common = pd.merge(
            df_prev,
            df_curr,
            on=["Svid", "SignalType"],
            suffixes=("_p", "_c"),
            how="inner",
        )

        if common.empty:
            return np.nan, np.nan, np.nan, np.nan, 0

        # Filter for valid ADR in BOTH epochs
        # We assume the input df has 'AccumulatedDeltaRangeState' and 'AccumulatedDeltaRangeMeters'
        valid_idx = common["AccumulatedDeltaRangeState_p"].apply(
            self._is_valid_adr
        ) & common["AccumulatedDeltaRangeState_c"].apply(self._is_valid_adr)
        common = common[valid_idx].copy()

        if len(common) < self.min_sats:
            return np.nan, np.nan, np.nan, np.nan, len(common)

        # Prepare Data for Least Squares
        # 1. Delta ADR (Measurement)
        # Note: ADR is in meters.
        delta_adr = (
            common["AccumulatedDeltaRangeMeters_c"]
            - common["AccumulatedDeltaRangeMeters_p"]
        )

        # 2. Satellite Positions
        # We need Sat position at current time (t) and prev time (t-1)
        sx_c = common["SvPositionXEcefMeters_c"].values
        sy_c = common["SvPositionYEcefMeters_c"].values
        sz_c = common["SvPositionZEcefMeters_c"].values

        sx_p = common["SvPositionXEcefMeters_p"].values
        sy_p = common["SvPositionYEcefMeters_p"].values
        sz_p = common["SvPositionZEcefMeters_p"].values

        # 3. Receiver Position (Linearization point)
        rx_x, rx_y, rx_z = xyz_prev

        # 4. Geometric Ranges from Previous Receiver Position
        # Range to Sat at t (current) from Rx at t-1
        dist_c = np.sqrt((sx_c - rx_x) ** 2 + (sy_c - rx_y) ** 2 + (sz_c - rx_z) ** 2)
        # Range to Sat at t-1 (prev) from Rx at t-1
        dist_p = np.sqrt((sx_p - rx_x) ** 2 + (sy_p - rx_y) ** 2 + (sz_p - rx_z) ** 2)

        # 5. Line of Sight Vector (Unit vector from Rx(t-1) to Sat(t))
        # This projects the displacement onto the satellite direction
        ux = (sx_c - rx_x) / dist_c
        uy = (sy_c - rx_y) / dist_c
        uz = (sz_c - rx_z) / dist_c

        # 6. Formulate Residual (y)
        # Equation: u * dx - c*dt = (dist_c - dist_p) - delta_adr
        # We define y = (dist_c - dist_p) - delta_adr
        # We solve H * [dx, dy, dz, c*dt]^T = y
        # Where H = [ux, uy, uz, -1]
        y = (dist_c - dist_p) - delta_adr

        # 7. Weights
        # Weight by Average Cn0
        avg_cn0 = (common["Cn0DbHz_c"] + common["Cn0DbHz_p"]) / 2.0
        weights = 10 ** (avg_cn0 / 10.0)
        W = np.diag(weights)

        # 8. Design Matrix H
        num_obs = len(y)
        H = np.zeros((num_obs, 4))
        H[:, 0] = ux
        H[:, 1] = uy
        H[:, 2] = uz
        H[:, 3] = -1.0

        # 9. Solve Weighted Least Squares
        # (H.T * W * H) x = H.T * W * y
        try:
            HTW = H.T @ W
            HTWH = HTW @ H
            HTWy = HTW @ y

            # Add small regularization for stability
            HTWH[np.diag_indices_from(HTWH)] += 1e-6

            x_sol = np.linalg.solve(HTWH, HTWy)

            return x_sol[0], x_sol[1], x_sol[2], x_sol[3], num_obs
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan, np.nan, num_obs

    def process_drive(self, df_gnss):
        """
        Process an entire drive to compute epoch-to-epoch displacements.

        Args:
            df_gnss (pd.DataFrame): Raw GNSS data for the drive.

        Returns:
            pd.DataFrame: DataFrame with columns [UnixTimeMillis, dx, dy, dz, dt_drift, tdcp_sats]
        """
        if df_gnss.empty:
            return pd.DataFrame(
                columns=["UnixTimeMillis", "dx", "dy", "dz", "dt_drift", "tdcp_sats"]
            )

        # Sort by time
        df_gnss = df_gnss.sort_values("utcTimeMillis").reset_index(drop=True)

        # Group by epoch
        # We create a list of (timestamp, dataframe) tuples
        epochs = list(df_gnss.groupby("utcTimeMillis"))

        results = []

        # Iterate through consecutive epochs
        for i in range(1, len(epochs)):
            ts_prev, df_prev = epochs[i - 1]
            ts_curr, df_curr = epochs[i]

            # Check time gap (max 1.5 seconds for 1Hz data)
            dt_ms = ts_curr - ts_prev
            if dt_ms > 1500:
                # Gap too large, cannot compute relative displacement
                results.append(
                    {
                        "UnixTimeMillis": ts_curr,
                        "dx": np.nan,
                        "dy": np.nan,
                        "dz": np.nan,
                        "dt_drift": np.nan,
                        "tdcp_sats": 0,
                    }
                )
                continue

            # Get Receiver Position at t-1 (Linearization point)
            # We use the WLS position provided in the dataset
            # Take the first valid WLS position in the previous epoch
            wls_x = df_prev["WlsPositionXEcefMeters"].iloc[0]
            wls_y = df_prev["WlsPositionYEcefMeters"].iloc[0]
            wls_z = df_prev["WlsPositionZEcefMeters"].iloc[0]

            if np.isnan(wls_x):
                # Fallback if WLS is missing (rare)
                results.append(
                    {
                        "UnixTimeMillis": ts_curr,
                        "dx": np.nan,
                        "dy": np.nan,
                        "dz": np.nan,
                        "dt_drift": np.nan,
                        "tdcp_sats": 0,
                    }
                )
                continue

            # Solve
            dx, dy, dz, dt, nsats = self.solve_step(
                df_prev, df_curr, (wls_x, wls_y, wls_z)
            )

            results.append(
                {
                    "UnixTimeMillis": ts_curr,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "dt_drift": dt,
                    "tdcp_sats": nsats,
                }
            )

        # Add the first epoch with NaNs (no previous history)
        if epochs:
            first_ts, _ = epochs[0]
            results.insert(
                0,
                {
                    "UnixTimeMillis": first_ts,
                    "dx": np.nan,
                    "dy": np.nan,
                    "dz": np.nan,
                    "dt_drift": np.nan,
                    "tdcp_sats": 0,
                },
            )

        return pd.DataFrame(results)


def get_tdcp_displacement(drive_id, phone_name, df_gnss, load_cached_data=True):
    """
    Get TDCP displacements for a specific drive, using caching.

    Args:
        drive_id (str): Drive ID.
        phone_name (str): Phone Name.
        df_gnss (pd.DataFrame): Raw GNSS data.
        load_cached_data (bool): Whether to use cache.

    Returns:
        pd.DataFrame: TDCP results.
    """
    cache_dir = Config.WORKING_DIR
    cache_file = os.path.join(cache_dir, f"tdcp_{drive_id}_{phone_name}.parquet")

    # 1. Try Load
    if load_cached_data and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass  # Fallback to compute

    # 2. Compute
    engine = TDCPEngine()
    tdcp_df = engine.process_drive(df_gnss)

    # 3. Save
    try:
        tdcp_df.to_parquet(cache_file, index=False)
    except Exception as e:
        print(f"Failed to cache TDCP for {drive_id}-{phone_name}: {e}")

    return tdcp_df
