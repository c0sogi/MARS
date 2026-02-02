import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy import sparse
from scipy.sparse.linalg import spsolve

from library.config import (
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    SEED,
    WORKING_DIR,
    SUBMISSION_FILE_PATH,
    HUBER_DELTA,
    WEIGHT_TDCP,
    WEIGHT_DOPPLER,
    SAMPLE_SUBMISSION_PATH,
)
from library.utils import enu_to_ecef, ecef_to_wgs84, ecef_to_enu
from library.data_manager import DataManager
from library.feature_builder import rotate_vector_ecef_to_enu


class SplitBandLGBM:
    def __init__(self):
        self.models_e = []
        self.models_n = []
        self.feature_names = []

    def _train_single_target(self, X, y, groups, target_name):
        print(f"Training models for {target_name}...")
        models = []
        gkf = GroupKFold(n_splits=5)

        # Metrics storage
        mae_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Create LightGBM datasets
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Silent
            ]

            model = lgb.train(
                LGBM_PARAMS,
                lgb_train,
                valid_sets=[lgb_val],
                valid_names=["valid"],
                num_boost_round=LGBM_PARAMS["n_estimators"],
                callbacks=callbacks,
            )

            # Evaluate
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            mae = np.mean(np.abs(y_val - y_pred))
            mae_scores.append(mae)

            models.append(model)
            print(f"  Fold {fold}: MAE = {mae:.9f}, Best Iter = {model.best_iteration}")

        print(f"Average MAE for {target_name}: {np.mean(mae_scores):.9f}")
        return models

    def train(self, train_df):
        """
        Train the East and North models using the provided training dataframe.
        """
        dm = DataManager()
        X = dm.get_X_y(train_df)
        self.feature_names = X.columns.tolist()

        # Targets
        y_e = train_df["target_E"]
        y_n = train_df["target_N"]
        groups = train_df["drive_id"]

        print(f"Training with {len(X)} samples, {len(self.feature_names)} features.")

        self.models_e = self._train_single_target(X, y_e, groups, "East")
        self.models_n = self._train_single_target(X, y_n, groups, "North")

    def _optimize_track_component(
        self, anchors, deltas, weights_kin, huber_delta=HUBER_DELTA
    ):
        """
        Optimize a 1D trajectory component (e.g., Easting) using IRLS.
        Cost = Sum( Huber(x_t - anchor_t) ) + Sum( w_kin * (x_t - x_{t-1} - delta_t)^2 )
        """
        n = len(anchors)
        if n == 0:
            return np.array([])

        # Initial guess: anchors
        x = anchors.copy()

        # IRLS parameters
        max_iter = 15
        tol = 1e-4

        for i in range(max_iter):
            x_prev = x.copy()

            # 1. Compute Anchor Weights based on Huber Loss
            # Huber: rho(r) = 0.5*r^2 if |r|<delta, else delta*(|r| - 0.5*delta)
            # Weight w = rho'(r)/r.
            # If |r| < delta, w = 1.
            # If |r| >= delta, w = delta / |r|.
            residuals = x - anchors
            abs_res = np.abs(residuals)

            # Avoid division by zero
            w_anchors = np.ones(n)
            mask_outlier = abs_res >= huber_delta
            w_anchors[mask_outlier] = huber_delta / abs_res[mask_outlier]

            # 2. Construct Sparse Linear System (Ax = b)
            # For each t:
            #   w_a_t * x_t + w_k_t * (x_t - x_{t-1}) - w_k_{t+1} * (x_{t+1} - x_t) = w_a_t * a_t + w_k_t * d_t - w_k_{t+1} * d_{t+1}

            # Diagonals
            # Main diagonal (coeff of x_t)
            diag_main = w_anchors.copy()
            diag_main[1:] += weights_kin[1:]
            diag_main[:-1] += weights_kin[1:]

            # Off-diagonals
            diag_upper = -weights_kin[1:]  # x_{t+1} coeff
            diag_lower = -weights_kin[1:]  # x_{t-1} coeff

            A = sparse.diags(
                [diag_lower, diag_main, diag_upper], [-1, 0, 1], format="csr"
            )

            # RHS Vector b
            b = w_anchors * anchors
            b[1:] += weights_kin[1:] * deltas[1:]
            b[:-1] -= weights_kin[1:] * deltas[1:]

            # Solve
            x = spsolve(A, b)

            # Check convergence
            if np.max(np.abs(x - x_prev)) < tol:
                break

        return x

    def _apply_graph_optimization(self, df):
        """
        Apply Multi-Modal Graph Optimization to a single trip.
        """
        # Sort by time
        df = df.sort_values("UnixTimeMillis").reset_index(drop=True)

        # 1. Extract Anchors (WLS + Predicted Residuals)
        # Note: Model predicts (GT - WLS). So GT = WLS + Pred.
        # Establish local tangent plane at first WLS point
        wls_x = df["WlsPositionXEcefMeters"].values
        wls_y = df["WlsPositionYEcefMeters"].values
        wls_z = df["WlsPositionZEcefMeters"].values

        # Reference: First point
        ref_x, ref_y, ref_z = wls_x[0], wls_y[0], wls_z[0]
        ref_lat, ref_lon, ref_alt = ecef_to_wgs84(ref_x, ref_y, ref_z)

        # Convert all WLS to ENU
        wls_e, wls_n, wls_u = ecef_to_enu(
            wls_x, wls_y, wls_z, ref_lat, ref_lon, ref_alt
        )

        # Add predicted residuals
        anchors_e = wls_e + df["pred_E"].values
        anchors_n = wls_n + df["pred_N"].values
        anchors_u = wls_u  # Keep vertical as WLS (or 0) since we don't predict it

        # 2. Extract Kinematics (Delta)
        # Time deltas
        t_diff = df["UnixTimeMillis"].diff().fillna(1000).values / 1000.0  # seconds

        # Prepare arrays
        delta_ecef_x = np.zeros(len(df))
        delta_ecef_y = np.zeros(len(df))
        delta_ecef_z = np.zeros(len(df))
        weights = np.zeros(len(df))

        # TDCP
        tdcp_valid = df["TDCP_Valid"].fillna(0).values.astype(bool)

        # Doppler fallback
        dop_vx = df["Doppler_Vel_X"].fillna(0).values
        dop_vy = df["Doppler_Vel_Y"].fillna(0).values
        dop_vz = df["Doppler_Vel_Z"].fillna(0).values

        # Fill Kinematics
        # TDCP
        delta_ecef_x[tdcp_valid] = df.loc[tdcp_valid, "TDCP_Disp_X"].values
        delta_ecef_y[tdcp_valid] = df.loc[tdcp_valid, "TDCP_Disp_Y"].values
        delta_ecef_z[tdcp_valid] = df.loc[tdcp_valid, "TDCP_Disp_Z"].values
        weights[tdcp_valid] = WEIGHT_TDCP

        # Doppler (where TDCP invalid)
        mask_dop = ~tdcp_valid
        delta_ecef_x[mask_dop] = dop_vx[mask_dop] * t_diff[mask_dop]
        delta_ecef_y[mask_dop] = dop_vy[mask_dop] * t_diff[mask_dop]
        delta_ecef_z[mask_dop] = dop_vz[mask_dop] * t_diff[mask_dop]
        weights[mask_dop] = WEIGHT_DOPPLER

        # First point has no incoming edge
        weights[0] = 0.0

        # Rotate ECEF deltas to ENU
        d_e, d_n, _ = rotate_vector_ecef_to_enu(
            delta_ecef_x, delta_ecef_y, delta_ecef_z, ref_lat, ref_lon
        )

        # 3. Optimize
        opt_e = self._optimize_track_component(anchors_e, d_e, weights)
        opt_n = self._optimize_track_component(anchors_n, d_n, weights)

        # 4. Convert back to Lat/Lon
        # ENU -> ECEF -> WGS84
        opt_x, opt_y, opt_z = enu_to_ecef(
            opt_e, opt_n, anchors_u, ref_lat, ref_lon, ref_alt
        )
        opt_lat, opt_lon, _ = ecef_to_wgs84(opt_x, opt_y, opt_z)

        df["LatitudeDegrees"] = opt_lat
        df["LongitudeDegrees"] = opt_lon

        return df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

    def predict(self, test_df):
        """
        Generate predictions for the test set.
        """
        dm = DataManager()
        X_test = dm.get_X_y(test_df)

        # Ensure features match training
        missing_feats = set(self.feature_names) - set(X_test.columns)
        if missing_feats:
            print(
                f"Warning: {len(missing_feats)} features missing in test set. Filling with 0."
            )
            for f in missing_feats:
                X_test[f] = 0

        X_test = X_test[self.feature_names]

        print("Predicting residuals...")
        # Average predictions from all fold models
        pred_e = np.zeros(len(X_test))
        for model in self.models_e:
            pred_e += model.predict(X_test)
        pred_e /= len(self.models_e)

        pred_n = np.zeros(len(X_test))
        for model in self.models_n:
            pred_n += model.predict(X_test)
        pred_n /= len(self.models_n)

        test_df["pred_E"] = pred_e
        test_df["pred_N"] = pred_n

        print("Running Graph Optimization per trip...")
        results = []
        for trip_id, group in test_df.groupby("tripId"):
            optimized_group = self._apply_graph_optimization(group.copy())
            results.append(optimized_group)

        submission = pd.concat(results, ignore_index=True)

        # Ensure correct order for submission
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        # We merge on tripId and UnixTimeMillis.
        # Note: sample submission might have different order or missing rows if we filtered.
        # But our test_df comes from test_metadata which comes from sample submission.

        # Left join to preserve sample submission structure
        submission = sample[["tripId", "UnixTimeMillis"]].merge(
            submission, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Fill any missing values (if any) with sample values or interpolation
        # (Should not happen if pipeline is correct)
        if submission["LatitudeDegrees"].isnull().any():
            print("Warning: NaNs in submission. Filling with linear interpolation.")
            submission["LatitudeDegrees"] = (
                submission["LatitudeDegrees"]
                .interpolate()
                .fillna(method="bfill")
                .fillna(method="ffill")
            )
            submission["LongitudeDegrees"] = (
                submission["LongitudeDegrees"]
                .interpolate()
                .fillna(method="bfill")
                .fillna(method="ffill")
            )

        # Save
        submission.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_FILE_PATH}")
        return submission
