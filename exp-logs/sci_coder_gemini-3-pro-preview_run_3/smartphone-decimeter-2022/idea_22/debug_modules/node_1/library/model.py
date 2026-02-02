import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import os
import gc

from library.config import LGBM_PARAMS, SEED, WORKING_DIR
from library.utils import wgs84_to_ecef, ecef_to_enu, enu_to_ecef, ecef_to_wgs84


class ResidualRegressor:
    def __init__(self):
        self.models_east = []
        self.models_north = []
        self.feature_cols = []

    def _calculate_targets(self, df):
        """
        Calculates ENU residuals (Ground Truth - WLS) for training.
        Handles missing GT Altitude by using WLS Altitude.
        """
        # Ensure GT columns exist
        if "LatitudeDegrees" not in df.columns or "LongitudeDegrees" not in df.columns:
            raise ValueError("Ground truth coordinates missing from training data.")

        # Handle missing GT Altitude
        # If GT Alt is missing, use WLS Alt. This projects GT Lat/Lon onto the estimated surface.
        # WLS Alt is derived from WLS ECEF.
        wls_x = df["WlsPositionXEcefMeters"].values
        wls_y = df["WlsPositionYEcefMeters"].values
        wls_z = df["WlsPositionZEcefMeters"].values

        # Get WLS LLA to serve as local origin and fallback altitude
        wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

        gt_lat = df["LatitudeDegrees"].values
        gt_lon = df["LongitudeDegrees"].values
        gt_alt = df["AltitudeMeters"].fillna(pd.Series(wls_alt)).values

        # Convert GT LLA to ECEF
        gt_x, gt_y, gt_z = wgs84_to_ecef(gt_lat, gt_lon, gt_alt)

        # Calculate ECEF Residuals (GT - WLS)
        dx = gt_x - wls_x
        dy = gt_y - wls_y
        dz = gt_z - wls_z

        # Rotate residuals to ENU frame centered at WLS position
        # We perform this manually vectorized or use the util.
        # The util ecef_to_enu takes absolute coords, so we can pass GT coords and WLS origin.
        e_res, n_res, u_res = ecef_to_enu(gt_x, gt_y, gt_z, wls_lat, wls_lon, wls_alt)

        return e_res, n_res

    def _preprocess_features(self, df, is_train=True):
        """
        Selects feature columns and handles basic cleaning.
        """
        # Identify feature columns (exclude IDs, timestamps, targets, and raw positions)
        exclude_cols = [
            "tripId",
            "UnixTimeMillis",
            "drive_id",
            "phone_name",
            "gt_path",
            "gnss_path",
            "imu_path",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "AltitudeMeters",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "target_E",
            "target_N",
        ]

        feats = [c for c in df.columns if c not in exclude_cols]

        # Basic NaN handling for features (LightGBM handles them, but infs can be issues)
        # We assume features are numeric.
        return feats

    def train(self, df_train, n_folds=5):
        """
        Trains the residual regressors using GroupKFold.

        Args:
            df_train (pd.DataFrame): Training data containing features and GT.
            n_folds (int): Number of cross-validation folds.
        """
        print(f"Preparing training data with {len(df_train)} samples...")

        # Calculate Targets
        target_e, target_n = self._calculate_targets(df_train)
        df_train["target_E"] = target_e
        df_train["target_N"] = target_n

        # Define Features
        self.feature_cols = self._preprocess_features(df_train)
        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

        X = df_train[self.feature_cols]
        y_e = df_train["target_E"]
        y_n = df_train["target_N"]
        groups = df_train["drive_id"]

        gkf = GroupKFold(n_splits=n_folds)

        # Reset models
        self.models_east = []
        self.models_north = []

        oof_e = np.zeros(len(df_train))
        oof_n = np.zeros(len(df_train))

        print(f"Starting {n_folds}-fold GroupKFold training...")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_e, groups)):
            print(f"\n--- Fold {fold + 1} ---")

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_e_tr, y_e_val = y_e.iloc[train_idx], y_e.iloc[val_idx]
            y_n_tr, y_n_val = y_n.iloc[train_idx], y_n.iloc[val_idx]

            # --- Train East Model ---
            print("Training East Model...")
            dtrain_e = lgb.Dataset(X_tr, label=y_e_tr)
            dval_e = lgb.Dataset(X_val, label=y_e_val, reference=dtrain_e)

            model_e = lgb.train(
                LGBM_PARAMS,
                dtrain_e,
                valid_sets=[dtrain_e, dval_e],
                valid_names=["train", "valid"],
                callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)],
            )
            self.models_east.append(model_e)
            oof_e[val_idx] = model_e.predict(
                X_val, num_iteration=model_e.best_iteration
            )

            # --- Train North Model ---
            print("Training North Model...")
            dtrain_n = lgb.Dataset(X_tr, label=y_n_tr)
            dval_n = lgb.Dataset(X_val, label=y_n_val, reference=dtrain_n)

            model_n = lgb.train(
                LGBM_PARAMS,
                dtrain_n,
                valid_sets=[dtrain_n, dval_n],
                valid_names=["train", "valid"],
                callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)],
            )
            self.models_north.append(model_n)
            oof_n[val_idx] = model_n.predict(
                X_val, num_iteration=model_n.best_iteration
            )

            # Fold Metrics
            mae_e = mean_absolute_error(y_e_val, oof_e[val_idx])
            mae_n = mean_absolute_error(y_n_val, oof_n[val_idx])
            print(f"Fold {fold+1} MAE - East: {mae_e:.6f}, North: {mae_n:.6f}")

            # Cleanup
            del (
                X_tr,
                X_val,
                y_e_tr,
                y_e_val,
                y_n_tr,
                y_n_val,
                dtrain_e,
                dval_e,
                dtrain_n,
                dval_n,
            )
            gc.collect()

        # Overall Metrics
        total_mae_e = mean_absolute_error(y_e, oof_e)
        total_mae_n = mean_absolute_error(y_n, oof_n)
        overall_mae = (total_mae_e + total_mae_n) / 2
        print(
            f"\nOverall OOF MAE - East: {total_mae_e:.6f}, North: {total_mae_n:.6f}, Avg: {overall_mae:.6f}"
        )

    def predict(self, df_test):
        """
        Generates absolute position predictions using the trained ensemble.

        Args:
            df_test (pd.DataFrame): Test data containing features and WLS positions.

        Returns:
            pd.DataFrame: DataFrame with ['tripId', 'UnixTimeMillis', 'lat_pred', 'lon_pred']
        """
        if not self.models_east or not self.models_north:
            raise RuntimeError("Models not trained. Call train() first.")

        print(f"Predicting on {len(df_test)} samples...")

        X_test = df_test[self.feature_cols]

        # Ensemble Prediction (Average)
        pred_e = np.zeros(len(df_test))
        pred_n = np.zeros(len(df_test))

        for model in self.models_east:
            pred_e += model.predict(X_test, num_iteration=model.best_iteration)
        pred_e /= len(self.models_east)

        for model in self.models_north:
            pred_n += model.predict(X_test, num_iteration=model.best_iteration)
        pred_n /= len(self.models_north)

        # Reconstruct Absolute Positions
        # 1. Get WLS Origin
        wls_x = df_test["WlsPositionXEcefMeters"].values
        wls_y = df_test["WlsPositionYEcefMeters"].values
        wls_z = df_test["WlsPositionZEcefMeters"].values

        wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

        # 2. Convert Predicted ENU Residuals to ECEF Delta
        # We assume Up residual is 0 for the anchor generation
        pred_u = np.zeros_like(pred_e)

        # enu_to_ecef returns absolute coordinates if we pass the origin
        # But our function returns the absolute ECEF directly.
        # Let's verify utils.py implementation:
        # x = x0 + dx ... yes, it returns absolute ECEF.
        pred_x, pred_y, pred_z = enu_to_ecef(
            pred_e, pred_n, pred_u, wls_lat, wls_lon, wls_alt
        )

        # 3. Convert Predicted ECEF to LLA
        pred_lat, pred_lon, _ = ecef_to_wgs84(pred_x, pred_y, pred_z)

        # Create Result DataFrame
        result = pd.DataFrame(
            {
                "tripId": df_test["tripId"],
                "UnixTimeMillis": df_test["UnixTimeMillis"],
                "lat_pred": pred_lat,
                "lon_pred": pred_lon,
            }
        )

        return result
