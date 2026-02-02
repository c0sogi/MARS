import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.config import (
    WEIGHT_ANCHOR,
    HUBER_DELTA,
    WORKING_DIR,
    SUBMISSION_PATH,
    INPUT_DIR,
)
from library.utils import (
    ecef_to_wgs84,
    wgs84_to_ecef,
    ecef_to_enu,
    enu_to_ecef,
    huber_loss,
)
from library.kinematics import compute_trajectory_deltas
from library.data_loader import load_drive_data


class TrajectoryOptimizer:
    """
    Implements Adaptive Factor Graph Optimization to smooth GNSS trajectories.
    Fuses ML-predicted anchors with Kinematic (TDCP/Doppler) constraints.
    """

    def __init__(self):
        self.weight_anchor = WEIGHT_ANCHOR
        self.huber_delta = HUBER_DELTA

    def _objective_function(self, x, anchors, deltas, weights):
        """
        Cost function for 1D optimization.
        J(x) = Sum(Huber(x - anchor)) + Sum(weight * (step - delta)^2)
        """
        # 1. Anchor Cost (Huber)
        residuals = x - anchors
        # Use library huber_loss (returns array), then sum
        loss_anchor = (
            np.sum(huber_loss(residuals, self.huber_delta)) * self.weight_anchor
        )

        # 2. Kinematic Cost (Weighted L2)
        # x[i] - x[i-1] should match delta[i]
        # deltas and weights are aligned such that index i corresponds to step (i-1 -> i)
        # We skip index 0 for kinematics as it has no previous point
        steps = x[1:] - x[:-1]
        kin_residuals = steps - deltas[1:]
        loss_kin = np.sum(weights[1:] * kin_residuals**2)

        return loss_anchor + loss_kin

    def _optimize_component(self, init_guess, anchors, deltas, weights):
        """
        Optimize a single dimension (East, North, or Up).
        """
        # Use L-BFGS-B for efficient bound-constrained (implicit) optimization
        # We don't set explicit bounds, but the method is robust for this scale
        result = minimize(
            fun=self._objective_function,
            x0=init_guess,
            args=(anchors, deltas, weights),
            method="L-BFGS-B",
            options={"disp": False, "maxiter": 10000},
        )
        return result.x

    def optimize_drive(
        self, drive_id, phone_name, metadata_df, ml_preds_df, load_cached_data=True
    ):
        """
        Optimize trajectory for a single drive.

        Args:
            drive_id (str): Drive Identifier.
            phone_name (str): Phone Model Name.
            metadata_df (pd.DataFrame): Metadata for file paths.
            ml_preds_df (pd.DataFrame): ML Predictions with columns ['UnixTimeMillis', 'pred_E', 'pred_N'].
            load_cached_data (bool): Whether to use cached optimization results.

        Returns:
            pd.DataFrame: Optimized trajectory with ['UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'].
        """
        # Cache Handling
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_file = f"opt_{drive_id}_{phone_name}.parquet"
        cache_path = os.path.join(WORKING_DIR, cache_file)

        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass  # Recompute on failure

        print(f"Optimizing trajectory for {drive_id} {phone_name}...")

        # 1. Load Data
        # Load aligned GNSS (WLS positions)
        gnss_df, _ = load_drive_data(
            drive_id, phone_name, metadata_df, load_cached_data=load_cached_data
        )

        # Load Kinematics (Deltas)
        kinematics_df = compute_trajectory_deltas(
            drive_id, phone_name, gnss_df, load_cached_data=load_cached_data
        )

        # 2. Merge Data Sources
        # Ensure we only work on common timestamps
        # ml_preds_df might cover all drives, filter it
        drive_preds = ml_preds_df[
            ml_preds_df["UnixTimeMillis"].isin(gnss_df["UnixTimeMillis"])
        ].copy()

        # Merge WLS, ML, and Kinematics
        # Base is GNSS (WLS)
        df = pd.merge(
            gnss_df[
                [
                    "UnixTimeMillis",
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ],
            drive_preds[["UnixTimeMillis", "pred_E", "pred_N"]],
            on="UnixTimeMillis",
            how="inner",
        )

        df = pd.merge(df, kinematics_df, on="UnixTimeMillis", how="inner")

        if df.empty:
            print(f"Warning: No overlapping data for {drive_id} {phone_name}")
            return pd.DataFrame()

        # Sort by time
        df = df.sort_values("UnixTimeMillis").reset_index(drop=True)

        # 3. Coordinate Transformation (ECEF -> Local ENU)
        # Define Origin: First WLS position
        x0 = df["WlsPositionXEcefMeters"].iloc[0]
        y0 = df["WlsPositionYEcefMeters"].iloc[0]
        z0 = df["WlsPositionZEcefMeters"].iloc[0]
        lat0, lon0, alt0 = ecef_to_wgs84(x0, y0, z0)

        # Convert WLS trajectory to ENU
        wls_e, wls_n, wls_u = ecef_to_enu(
            df["WlsPositionXEcefMeters"].values,
            df["WlsPositionYEcefMeters"].values,
            df["WlsPositionZEcefMeters"].values,
            lat0,
            lon0,
            alt0,
        )

        # Apply ML Predictions (Residuals) to get Anchors
        # Anchor = WLS + Residual
        # Note: If ML only predicts E/N, we assume U residual is 0 for anchor (trust WLS altitude)
        anchors_e = wls_e + df["pred_E"].values
        anchors_n = wls_n + df["pred_N"].values
        anchors_u = wls_u  # No ML correction for Up

        # Convert Kinematic Deltas (ECEF) to ENU
        # Delta_ENU = R * Delta_ECEF
        # We use ecef_to_enu on (Origin + Delta) to get rotated delta
        k_dx = df["dx"].values
        k_dy = df["dy"].values
        k_dz = df["dz"].values

        # Vectorized rotation via utils function trick
        # Pass (x0+dx, y0+dy, z0+dz) as target, (lat0, lon0, alt0) as ref
        # Result is (e, n, u) which represents the delta in ENU
        d_e, d_n, d_u = ecef_to_enu(x0 + k_dx, y0 + k_dy, z0 + k_dz, lat0, lon0, alt0)

        weights = df["weight"].values

        # 4. Optimization
        # Optimize East
        opt_e = self._optimize_component(anchors_e, anchors_e, d_e, weights)
        # Optimize North
        opt_n = self._optimize_component(anchors_n, anchors_n, d_n, weights)
        # Optimize Up (Optional, but good for consistency)
        opt_u = self._optimize_component(anchors_u, anchors_u, d_u, weights)

        # 5. Convert back to WGS84
        opt_x, opt_y, opt_z = enu_to_ecef(opt_e, opt_n, opt_u, lat0, lon0, alt0)
        opt_lat, opt_lon, _ = ecef_to_wgs84(opt_x, opt_y, opt_z)

        # 6. Format Result
        result_df = pd.DataFrame(
            {
                "tripId": f"{drive_id}-{phone_name}",
                "UnixTimeMillis": df["UnixTimeMillis"],
                "LatitudeDegrees": opt_lat,
                "LongitudeDegrees": opt_lon,
            }
        )

        # Save Cache
        result_df.to_parquet(cache_path, index=False)

        return result_df


def generate_submission(test_metadata_df, ml_predictions_df, load_cached_data=True):
    """
    Generate the final submission file by optimizing all test drives.

    Args:
        test_metadata_df (pd.DataFrame): Metadata for test set.
        ml_predictions_df (pd.DataFrame): ML predictions for test set.
        load_cached_data (bool): Whether to use cached intermediate files.
    """
    optimizer = TrajectoryOptimizer()
    results = []

    # Get unique drives in test set
    drive_phones = test_metadata_df[["drive_id", "phone_name"]].drop_duplicates().values

    print(f"Generating submission for {len(drive_phones)} traces...")

    for drive_id, phone_name in drive_phones:
        try:
            df = optimizer.optimize_drive(
                drive_id,
                phone_name,
                test_metadata_df,
                ml_predictions_df,
                load_cached_data=load_cached_data,
            )
            if not df.empty:
                results.append(df)
        except Exception as e:
            print(f"Error optimizing {drive_id} {phone_name}: {e}")

    if not results:
        print("Error: No results generated!")
        return

    # Concatenate all results
    full_submission = pd.concat(results, ignore_index=True)

    # Ensure format matches sample submission
    # Load sample to get correct order/rows
    sample = pd.read_csv(
        SUBMISSION_PATH.replace("submission.csv", "../input/sample_submission.csv")
    )

    # Merge predictions onto sample
    # We use left join on tripId and UnixTimeMillis to ensure we fill the exact rows required
    output = sample[["tripId", "UnixTimeMillis"]].merge(
        full_submission[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ],
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    # Fill missing values (if any) with sample values or interpolation
    # In a robust pipeline, we might fallback to WLS, but here we assume optimization worked
    # If NaN, fill with 0 or forward fill (though optimization shouldn't produce NaNs)
    if output.isnull().any().any():
        print("Warning: NaNs in submission. Filling with linear interpolation.")
        output = output.interpolate().fillna(method="bfill").fillna(method="ffill")

    # Save
    output.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
