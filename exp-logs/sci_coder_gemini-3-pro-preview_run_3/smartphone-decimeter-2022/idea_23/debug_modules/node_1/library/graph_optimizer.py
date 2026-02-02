import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm
from library.config import WORKING_DIR, HUBER_DELTA, ANCHOR_WEIGHT, SUBMISSION_DIR
from library.coord_utils import get_enu_rotation_matrix


# Helper for ECEF to WGS84 conversion
def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Latitude/Longitude.
    Uses Ferrari's solution or iterative method.
    """
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b**2 / a**2)
    ep2 = (a**2 / b**2) - 1

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    return np.degrees(lat), np.degrees(lon)


class GraphOptimizer:
    """
    Fuses ML Anchors and Hybrid Kinematics using Global Optimization.
    Minimizes Cost J(X) = Sum(Huber(X - Anchor)) + Sum(w * ||dX - Kinematics||^2)
    """

    def __init__(self, working_dir=WORKING_DIR):
        self.working_dir = working_dir
        self.cache_dir = os.path.join(working_dir, "opt_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, drive_id, phone_name):
        return os.path.join(self.cache_dir, f"opt_{drive_id}_{phone_name}.parquet")

    def _residuals(self, x_flat, anchor_pos, kin_disp, kin_weights, n_samples):
        """
        Computes the residual vector for least squares.
        x_flat: Flattened state vector (3 * N,)
        """
        X = x_flat.reshape((n_samples, 3))

        # 1. Anchor Residuals: sqrt(w) * (X - P_anchor)
        # ANCHOR_WEIGHT is scalar
        res_anchor = (X - anchor_pos) * np.sqrt(ANCHOR_WEIGHT)

        # 2. Kinematic Residuals: sqrt(w) * ((X_t - X_{t-1}) - Delta_kin)
        # X[1:] is X_t, X[:-1] is X_{t-1}
        delta_X = X[1:] - X[:-1]
        diff = delta_X - kin_disp

        # kin_weights is (N-1,) -> broadcast to (N-1, 3)
        w_kin = np.sqrt(kin_weights)[:, np.newaxis]
        res_kin = diff * w_kin

        # Flatten and concatenate
        return np.concatenate([res_anchor.ravel(), res_kin.ravel()])

    def solve_trajectory(
        self, drive_id, phone_name, anchor_df, kinematics_df, load_cached_data=True
    ):
        """
        Optimizes the trajectory for a single drive.

        Args:
            drive_id (str): Drive ID.
            phone_name (str): Phone name.
            anchor_df (pd.DataFrame): Dataframe with 'UnixTimeMillis', 'WlsPositionXEcefMeters', ... 'Pred_E', 'Pred_N'.
            kinematics_df (pd.DataFrame): Dataframe with 'UnixTimeMillis', 'dx', 'dy', 'dz', 'weight'.
            load_cached_data (bool): Whether to use caching.

        Returns:
            pd.DataFrame: Optimized trajectory with 'tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'.
        """
        cache_path = self._get_cache_path(drive_id, phone_name)

        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        # Merge Data
        # Ensure we keep all anchors, merge kinematics where available
        df = pd.merge(
            anchor_df,
            kinematics_df,
            on="UnixTimeMillis",
            how="left",
            suffixes=("", "_kin"),
        )

        # Fill missing kinematics with 0 displacement and 0 weight (break chain)
        df["dx"] = df["dx"].fillna(0.0)
        df["dy"] = df["dy"].fillna(0.0)
        df["dz"] = df["dz"].fillna(0.0)
        df["weight"] = df["weight"].fillna(0.0)

        df = df.sort_values("UnixTimeMillis").reset_index(drop=True)
        n_samples = len(df)

        if n_samples == 0:
            return pd.DataFrame()

        # Prepare Anchor Positions (ECEF)
        wls_pos = df[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].values

        # Get Predictions (default to 0 if missing)
        pred_e = df["Pred_E"].values if "Pred_E" in df.columns else np.zeros(n_samples)
        pred_n = df["Pred_N"].values if "Pred_N" in df.columns else np.zeros(n_samples)
        pred_u = np.zeros(n_samples)  # Assume 0 Up correction

        # Get Lat/Lon for rotation
        if "Wls_Lat" in df.columns and "Wls_Lon" in df.columns:
            lats = df["Wls_Lat"].values
            lons = df["Wls_Lon"].values
        else:
            # Fallback: Compute rough Lat/Lon from WLS ECEF
            lats, lons = ecef_to_wgs84(wls_pos[:, 0], wls_pos[:, 1], wls_pos[:, 2])

        # Convert ENU predictions to ECEF offsets
        anchor_pos = np.zeros_like(wls_pos)
        for i in range(n_samples):
            R = get_enu_rotation_matrix(lats[i], lons[i])
            d_enu = np.array([pred_e[i], pred_n[i], pred_u[i]])
            # R is ECEF->ENU. Transpose is ENU->ECEF.
            d_ecef = R.T @ d_enu
            anchor_pos[i] = wls_pos[i] + d_ecef

        # Prepare Kinematics (N-1)
        # Row i in df corresponds to time t_i.
        # Kinematics at row i represents displacement from t_{i-1} to t_i.
        # The first row (i=0) has no previous, so kinematics are usually init/0.
        # We slice from index 1 to end.

        kin_disp = df[["dx", "dy", "dz"]].iloc[1:].values
        kin_weights = df["weight"].iloc[1:].values

        # Optimization
        x0 = anchor_pos.flatten()

        # Run Least Squares
        # Using 'trf' solver which handles sparse problems well implicitly or explicitly.
        # With Huber loss for robustness against anchor outliers.
        res = least_squares(
            self._residuals,
            x0,
            args=(anchor_pos, kin_disp, kin_weights, n_samples),
            loss="huber",
            f_scale=HUBER_DELTA,
            verbose=0,
            max_nfev=50,  # Limit iterations for speed
        )

        # Extract results
        x_opt = res.x.reshape((n_samples, 3))

        # Convert to Lat/Lon
        opt_lats, opt_lons = ecef_to_wgs84(x_opt[:, 0], x_opt[:, 1], x_opt[:, 2])

        # Build Result DataFrame
        result = df[["tripId", "UnixTimeMillis"]].copy()
        result["LatitudeDegrees"] = opt_lats
        result["LongitudeDegrees"] = opt_lons

        # Save cache
        try:
            result.to_parquet(cache_path, index=False)
        except Exception:
            pass

        return result


def generate_submission(
    test_metadata,
    feature_df,
    kinematics_engine,
    model_wrapper,
    output_path=os.path.join(SUBMISSION_DIR, "submission.csv"),
):
    """
    Runs the full inference pipeline and generates submission file.
    """
    print("Generating submission...")

    # 1. Predict Anchors
    print("Predicting anchors...")
    pred_e, pred_n = model_wrapper.predict(feature_df)

    # Add predictions to feature_df
    # We work on a copy to avoid side effects
    aug_features = feature_df.copy()
    aug_features["Pred_E"] = pred_e
    aug_features["Pred_N"] = pred_n

    # 2. Optimize per Trip
    optimizer = GraphOptimizer()

    # Get unique trips from metadata
    unique_trips = test_metadata[["tripId", "drive_id", "phone_name"]].drop_duplicates()

    results = []

    # Import loader locally to avoid circular imports if any
    from library.data_loader import GnssLoader

    loader = GnssLoader()

    for _, row in tqdm(
        unique_trips.iterrows(), total=len(unique_trips), desc="Optimizing Trips"
    ):
        trip_id = row["tripId"]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        # Filter features for this trip
        trip_anchors = aug_features[aug_features["tripId"] == trip_id].copy()

        if trip_anchors.empty:
            print(f"Warning: No features for {trip_id}")
            continue

        try:
            # Load raw GNSS for kinematics
            gnss_df, _, _ = loader.get_drive_data(drive_id, phone_name, split="test")

            # Compute Kinematics
            kin_df = kinematics_engine.compute_displacements(
                gnss_df, drive_id, phone_name
            )

            # Optimize
            trip_res = optimizer.solve_trajectory(
                drive_id, phone_name, trip_anchors, kin_df
            )

            if not trip_res.empty:
                results.append(trip_res)

        except Exception as e:
            print(f"Error processing {trip_id}: {e}")

    # 3. Concatenate and Format
    if not results:
        print("No results generated!")
        return

    final_df = pd.concat(results, ignore_index=True)

    # Ensure correct columns and format
    submission_cols = [
        "tripId",
        "UnixTimeMillis",
        "LatitudeDegrees",
        "LongitudeDegrees",
    ]

    # Filter to match sample submission rows exactly if needed, but usually sample submission defines the rows.
    # We should merge with sample submission to ensure exact row match.
    sample_sub = pd.read_csv(
        os.path.join(WORKING_DIR, "../input/sample_submission.csv")
    )

    # Merge predictions into sample submission
    # We use left join on sample submission to ensure we have all required rows
    out_df = sample_sub[["tripId", "UnixTimeMillis"]].merge(
        final_df, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Fill missing values (if any) with original sample values (or interpolation)
    # Here we just fill with WLS from sample if optimization failed for some points
    # But sample submission has dummy values.
    # Ideally we should have predictions for all. If missing, we might forward fill.
    out_df["LatitudeDegrees"] = (
        out_df["LatitudeDegrees"].fillna(method="ffill").fillna(method="bfill")
    )
    out_df["LongitudeDegrees"] = (
        out_df["LongitudeDegrees"].fillna(method="ffill").fillna(method="bfill")
    )

    # Save
    out_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
