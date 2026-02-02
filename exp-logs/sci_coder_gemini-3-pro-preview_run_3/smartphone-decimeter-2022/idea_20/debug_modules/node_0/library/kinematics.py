import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RANSACRegressor
from library.utils import CoordinateTransformer, IOHelper


class KinematicsEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.doppler_threshold = self.config.get("doppler_ransac_threshold", 2.0)  # m/s
        self.tdcp_threshold = self.config.get("tdcp_ransac_threshold", 0.05)  # meters
        self.min_samples = self.config.get("min_samples", 5)
        self.cache_dir = "./working/idea_20/"

        # Constants
        self.CLIGHT = 299792458.0

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_los_vectors(self, df):
        """
        Compute Line of Sight vectors from Receiver (WLS) to Satellites.
        """
        dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
        dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
        dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]

        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        df["ux"] = dx / dist
        df["uy"] = dy / dist
        df["uz"] = dz / dist
        df["range_sat_rec"] = dist
        return df

    def _solve_ransac(self, H, y, threshold):
        """
        Solve linear system y = Hx using RANSAC.
        """
        if len(y) < self.min_samples:
            return np.zeros(3), 0, False

        # RANSAC for x, y, z, clock_drift (4 states)
        # We only care about x, y, z
        model = RANSACRegressor(
            min_samples=self.min_samples, residual_threshold=threshold, random_state=42
        )
        try:
            model.fit(H, y)
            # Coefficients are the velocity/displacement vector + clock term
            # model.estimator_.coef_ might be shape (4,) or (1, 4) depending on sklearn version/y shape
            # If y is (N,), coef is (4,). If y is (N, 1), coef is (1, 4).

            # Standard LinearRegression in RANSAC fits y = Xw + b.
            # We construct H with 1s column for clock bias if we want intercept handled explicitly,
            # or rely on fit_intercept=True.
            # GNSS equations: rho_rate = u*v + clk_drift.
            # H contains [ux, uy, uz]. We let RANSAC handle the intercept (clock drift).

            est = model.estimator_
            vel = est.coef_  # [vx, vy, vz]

            # Check for validity
            valid = np.sum(model.inlier_mask_) >= self.min_samples
            return vel, np.sum(model.inlier_mask_), valid
        except Exception:
            return np.zeros(3), 0, False

    def _compute_doppler_epoch(self, group):
        """
        Compute Doppler velocity for a single epoch.
        """
        # Filter good measurements
        # PseudorangeRateUncertaintyMetersPerSecond < threshold?
        # Let's trust RANSAC to filter bad ones, but remove NaNs
        group = group.dropna(
            subset=[
                "PseudorangeRateMetersPerSecond",
                "ux",
                "uy",
                "uz",
                "SvVelocityXEcefMetersPerSecond",
                "SvVelocityYEcefMetersPerSecond",
                "SvVelocityZEcefMetersPerSecond",
            ]
        )

        if len(group) < self.min_samples:
            return pd.Series(
                {"vx": 0.0, "vy": 0.0, "vz": 0.0, "n_inliers": 0, "valid": False}
            )

        # Equation: PRR = u * (v_sat - v_rec) + clk_drift
        # PRR - u * v_sat = - u * v_rec + clk_drift
        # y = PRR - (ux*vsx + uy*vsy + uz*vsz)
        # H = - [ux, uy, uz]

        u = group[["ux", "uy", "uz"]].values
        v_sat = group[
            [
                "SvVelocityXEcefMetersPerSecond",
                "SvVelocityYEcefMetersPerSecond",
                "SvVelocityZEcefMetersPerSecond",
            ]
        ].values
        prr = group["PseudorangeRateMetersPerSecond"].values

        # Project sat velocity onto LOS
        sat_proj = np.sum(u * v_sat, axis=1)

        # Target for linear system
        y = prr - sat_proj

        # Design matrix (negative LOS for receiver velocity)
        H = -u

        vel, n_inliers, valid = self._solve_ransac(H, y, self.doppler_threshold)

        return pd.Series(
            {
                "vx": vel[0],
                "vy": vel[1],
                "vz": vel[2],
                "n_inliers": n_inliers,
                "valid": valid,
            }
        )

    def _compute_tdcp_deltas(self, df):
        """
        Compute TDCP displacement between consecutive epochs.
        """
        # Create unique signal identifier
        df["SignalID"] = df["Svid"].astype(str) + "_" + df["SignalType"].astype(str)

        # Sort by signal and time
        df = df.sort_values(["SignalID", "utcTimeMillis"])

        # Shift to get previous epoch data
        cols_to_shift = [
            "utcTimeMillis",
            "AccumulatedDeltaRangeMeters",
            "AccumulatedDeltaRangeState",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        shifted = df.groupby("SignalID")[cols_to_shift].shift(1)

        # Rename shifted columns
        shifted.columns = [f"prev_{c}" for c in shifted.columns]

        # Join back
        df = pd.concat([df, shifted], axis=1)

        # Filter for consecutive epochs (allow small gap, e.g. < 1.1s)
        # Timestamps are millis
        time_diff = df["utcTimeMillis"] - df["prev_utcTimeMillis"]
        valid_time = (time_diff > 0) & (time_diff < 1100)

        # Filter for valid ADR state
        # Bit 0: Valid, Bit 1: Reset, Bit 2: Cycle Slip
        # We need Valid=1, Reset=0, CycleSlip=0
        # State & 1 == 1, State & 2 == 0, State & 4 == 0
        # Actually, we need to check if a cycle slip happened *between* t-1 and t.
        # The 'AccumulatedDeltaRangeState' at time t indicates the state of the measurement at time t.
        # If Reset or CycleSlip is set at time t, we cannot use the difference (ADR_t - ADR_t-1).

        # Check current state
        state = df["AccumulatedDeltaRangeState"]
        valid_adr = ((state & 1) == 1) & ((state & 2) == 0) & ((state & 4) == 0)

        # Apply filters
        df_tdcp = df[valid_time & valid_adr].copy()

        if df_tdcp.empty:
            return pd.DataFrame()

        # Calculate Delta ADR
        # Note: Android raw ADR is usually -Phase * Lambda.
        # Increasing range (sat moving away) -> Phase decreases -> ADR decreases?
        # Standard convention: ADR = - Range + Clock + Ambiguity.
        # Delta ADR = - Delta Range + Delta Clock.
        # We will solve for Delta Range.
        # Delta Range = Range(t) - Range(t-1)
        # Range(t) approx ||Sat(t) - Rec(t-1)|| - u(t) * Delta_Rec
        # Range(t-1) = ||Sat(t-1) - Rec(t-1)||
        # Delta Range approx ||Sat(t) - Rec(t-1)|| - ||Sat(t-1) - Rec(t-1)|| - u(t) * Delta_Rec
        # Delta ADR = - (GeometryChange - u(t) * Delta_Rec) + Delta Clock
        # Delta ADR = - GeometryChange + u(t) * Delta_Rec + Delta Clock
        # Target y = Delta ADR + GeometryChange
        # Solve y = u(t) * Delta_Rec + Delta Clock

        # Previous receiver position (WLS at t-1)
        rx_prev_x = df_tdcp["prev_WlsPositionXEcefMeters"]
        rx_prev_y = df_tdcp["prev_WlsPositionYEcefMeters"]
        rx_prev_z = df_tdcp["prev_WlsPositionZEcefMeters"]

        # Sat positions
        sx_t, sy_t, sz_t = (
            df_tdcp["SvPositionXEcefMeters"],
            df_tdcp["SvPositionYEcefMeters"],
            df_tdcp["SvPositionZEcefMeters"],
        )
        sx_p, sy_p, sz_p = (
            df_tdcp["prev_SvPositionXEcefMeters"],
            df_tdcp["prev_SvPositionYEcefMeters"],
            df_tdcp["prev_SvPositionZEcefMeters"],
        )

        # Distances from Rec(t-1)
        dist_t = np.sqrt(
            (sx_t - rx_prev_x) ** 2 + (sy_t - rx_prev_y) ** 2 + (sz_t - rx_prev_z) ** 2
        )
        dist_p = np.sqrt(
            (sx_p - rx_prev_x) ** 2 + (sy_p - rx_prev_y) ** 2 + (sz_p - rx_prev_z) ** 2
        )

        geometry_change = dist_t - dist_p

        # Delta ADR
        delta_adr = (
            df_tdcp["AccumulatedDeltaRangeMeters"]
            - df_tdcp["prev_AccumulatedDeltaRangeMeters"]
        )

        # Target vector
        # Assuming ADR = -Range. If ADR = +Range, then y = Delta ADR - GeometryChange.
        # We will assume standard Android definition (ADR = -Range) initially.
        # y = Delta ADR + GeometryChange
        df_tdcp["y_tdcp"] = delta_adr + geometry_change

        # LOS vectors at t (already computed as ux, uy, uz in preprocessing)
        # H = [ux, uy, uz]

        return df_tdcp

    def _solve_tdcp_epoch(self, group):
        """
        Solve TDCP for a single epoch.
        """
        if len(group) < self.min_samples:
            return pd.Series(
                {"dx": 0.0, "dy": 0.0, "dz": 0.0, "n_inliers": 0, "valid": False}
            )

        H = group[["ux", "uy", "uz"]].values
        y = group["y_tdcp"].values

        disp, n_inliers, valid = self._solve_ransac(H, y, self.tdcp_threshold)

        return pd.Series(
            {
                "dx": disp[0],
                "dy": disp[1],
                "dz": disp[2],
                "n_inliers": n_inliers,
                "valid": valid,
            }
        )

    def process_trip(self, df_gnss, load_cached_data=True):
        """
        Process a single trip to extract kinematic constraints.
        """
        trip_id = df_gnss["tripId"].iloc[0]
        cache_file = f"kinematics_{trip_id}.parquet"

        # 1. Try Load Cache
        if load_cached_data:
            cached_df = IOHelper.load_parquet(cache_file)
            if cached_df is not None:
                return cached_df

        # 2. Preprocessing
        # Compute LOS vectors
        df_gnss = self._compute_los_vectors(df_gnss)

        # 3. Doppler Velocity
        # Group by epoch
        doppler_results = df_gnss.groupby("utcTimeMillis").apply(
            self._compute_doppler_epoch
        )
        doppler_results = doppler_results.reset_index()
        doppler_results.rename(
            columns={
                "vx": "v_x_dop",
                "vy": "v_y_dop",
                "vz": "v_z_dop",
                "n_inliers": "n_dop",
                "valid": "valid_dop",
            },
            inplace=True,
        )

        # 4. TDCP Displacement
        df_tdcp_prep = self._compute_tdcp_deltas(df_gnss)

        if not df_tdcp_prep.empty:
            tdcp_results = df_tdcp_prep.groupby("utcTimeMillis").apply(
                self._solve_tdcp_epoch
            )
            tdcp_results = tdcp_results.reset_index()
            tdcp_results.rename(
                columns={
                    "dx": "d_x_tdcp",
                    "dy": "d_y_tdcp",
                    "dz": "d_z_tdcp",
                    "n_inliers": "n_tdcp",
                    "valid": "valid_tdcp",
                },
                inplace=True,
            )
        else:
            # Create empty structure if no TDCP possible
            tdcp_results = pd.DataFrame(
                columns=[
                    "utcTimeMillis",
                    "d_x_tdcp",
                    "d_y_tdcp",
                    "d_z_tdcp",
                    "n_tdcp",
                    "valid_tdcp",
                ]
            )

        # 5. Merge Results
        # Base is unique epochs
        epochs = df_gnss[
            [
                "utcTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].drop_duplicates("utcTimeMillis")

        merged = pd.merge(epochs, doppler_results, on="utcTimeMillis", how="left")
        merged = pd.merge(merged, tdcp_results, on="utcTimeMillis", how="left")

        # Fill NaNs
        merged[["v_x_dop", "v_y_dop", "v_z_dop"]] = merged[
            ["v_x_dop", "v_y_dop", "v_z_dop"]
        ].fillna(0.0)
        merged[["d_x_tdcp", "d_y_tdcp", "d_z_tdcp"]] = merged[
            ["d_x_tdcp", "d_y_tdcp", "d_z_tdcp"]
        ].fillna(0.0)
        merged["valid_dop"] = merged["valid_dop"].fillna(False)
        merged["valid_tdcp"] = merged["valid_tdcp"].fillna(False)

        # 6. Convert to ENU
        # We need a reference point. We can use the first WLS position of the trip as the local origin for ENU conversion
        # Or convert deltas directly using the rotation matrix at each epoch.
        # Since these are small deltas, we can rotate them using the current WLS position lat/lon.

        # Get Lat/Lon for rotation matrix
        wgs_x, wgs_y, wgs_z = (
            merged["WlsPositionXEcefMeters"].values,
            merged["WlsPositionYEcefMeters"].values,
            merged["WlsPositionZEcefMeters"].values,
        )
        lat, lon, _ = CoordinateTransformer.ecef_to_wgs84(wgs_x, wgs_y, wgs_z)

        # Rotate Doppler Velocity
        # Note: ecef_to_enu function converts positions. For vectors, we need just the rotation.
        # We can use ecef_to_enu with ref point = (0,0,0) and target = (vx, vy, vz) IF we adjust the logic,
        # but simpler to just implement vector rotation here or reuse the static method logic.
        # The static method ecef_to_enu does: dx = x - ref_x. If ref_x=0, dx=vx.
        # So we can pass (vx, vy, vz) and (lat, lon, 0) as ref, but set ref_xyz to 0 inside? No.
        # Let's just manually rotate.

        sin_lat = np.sin(np.radians(lat))
        cos_lat = np.cos(np.radians(lat))
        sin_lon = np.sin(np.radians(lon))
        cos_lon = np.cos(np.radians(lon))

        def rotate_vec(vx, vy, vz, sl, cl, slo, clo):
            e = -slo * vx + clo * vy
            n = -sl * clo * vx - sl * slo * vy + cl * vz
            u = cl * clo * vx + cl * slo * vy + sl * vz
            return e, n, u

        # Doppler ENU
        ve_d, vn_d, vu_d = rotate_vec(
            merged["v_x_dop"],
            merged["v_y_dop"],
            merged["v_z_dop"],
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
        )
        merged["v_east_dop"] = ve_d
        merged["v_north_dop"] = vn_d
        merged["v_up_dop"] = vu_d

        # TDCP ENU
        de_t, dn_t, du_t = rotate_vec(
            merged["d_x_tdcp"],
            merged["d_y_tdcp"],
            merged["d_z_tdcp"],
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
        )
        merged["d_east_tdcp"] = de_t
        merged["d_north_tdcp"] = dn_t
        merged["d_up_tdcp"] = du_t

        # 7. Sign Correction Check
        # If TDCP and Doppler are valid, they should align.
        # Dot product > 0. If < 0, TDCP sign might be flipped due to ADR definition.
        # Check median alignment
        valid_both = merged["valid_dop"] & merged["valid_tdcp"]
        if valid_both.sum() > 10:
            dot_prod = (
                merged.loc[valid_both, "v_east_dop"]
                * merged.loc[valid_both, "d_east_tdcp"]
                + merged.loc[valid_both, "v_north_dop"]
                * merged.loc[valid_both, "d_north_tdcp"]
            )

            # If correlation is negative, flip TDCP
            if dot_prod.median() < 0:
                print(f"  Flipping TDCP sign for trip {trip_id}")
                merged["d_east_tdcp"] *= -1
                merged["d_north_tdcp"] *= -1
                merged["d_up_tdcp"] *= -1
                merged["d_x_tdcp"] *= -1
                merged["d_y_tdcp"] *= -1
                merged["d_z_tdcp"] *= -1

        # Select output columns
        output_cols = [
            "utcTimeMillis",
            "tripId",
            "v_east_dop",
            "v_north_dop",
            "v_up_dop",
            "valid_dop",
            "n_dop",
            "d_east_tdcp",
            "d_north_tdcp",
            "d_up_tdcp",
            "valid_tdcp",
            "n_tdcp",
        ]

        final_df = merged[output_cols]

        # Save Cache
        IOHelper.save_parquet(final_df, cache_file)

        return final_df
