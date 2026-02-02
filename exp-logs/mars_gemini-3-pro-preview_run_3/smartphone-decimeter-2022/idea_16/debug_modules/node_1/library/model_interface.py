import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import ECEF_to_WGS84, ENU_to_WGS84


class ResidualRegressor:
    """
    Regressor for GNSS ENU Residuals using LightGBM.
    Trains separate models for East and North components.
    Uses GroupKFold to prevent data leakage across drives.
    """

    def __init__(self):
        self.feature_cols = Config.FEATURE_COLUMNS
        self.params = Config.LGBM_PARAMS
        self.models_E = []
        self.models_N = []
        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model_paths(self, fold):
        """Get file paths for saving/loading models for a specific fold."""
        path_e = os.path.join(self.model_dir, f"lgbm_E_fold_{fold}.txt")
        path_n = os.path.join(self.model_dir, f"lgbm_N_fold_{fold}.txt")
        return path_e, path_n

    def train(self, df, force_retrain=False):
        """
        Train the ensemble of LightGBM models.

        Args:
            df (pd.DataFrame): Training data containing features and targets.
            force_retrain (bool): If True, ignore cache and retrain.
        """
        X = df[self.feature_cols]
        y_E = df["res_E"]
        y_N = df["res_N"]
        groups = df["drive_id"]

        gkf = GroupKFold(n_splits=Config.NUM_FOLDS)

        # Check if all models exist in cache
        all_cached = True
        if not force_retrain:
            for i in range(Config.NUM_FOLDS):
                p_e, p_n = self._get_model_paths(i)
                if not (os.path.exists(p_e) and os.path.exists(p_n)):
                    all_cached = False
                    break

        if all_cached and not force_retrain:
            print("Loading trained models from cache...")
            self.models_E = []
            self.models_N = []
            for i in range(Config.NUM_FOLDS):
                p_e, p_n = self._get_model_paths(i)
                self.models_E.append(lgb.Booster(model_file=p_e))
                self.models_N.append(lgb.Booster(model_file=p_n))
            return

        print(
            f"Training ResidualRegressor on {len(df)} samples with {len(self.feature_cols)} features..."
        )
        self.models_E = []
        self.models_N = []

        mae_E_scores = []
        mae_N_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_E, groups)):
            print(f"\n--- Fold {fold + 1}/{Config.NUM_FOLDS} ---")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

            # --- Train East Model ---
            y_train_E, y_val_E = y_E.iloc[train_idx], y_E.iloc[val_idx]
            d_train_E = lgb.Dataset(X_train, label=y_train_E)
            d_val_E = lgb.Dataset(X_val, label=y_val_E, reference=d_train_E)

            model_E = lgb.train(
                self.params,
                d_train_E,
                valid_sets=[d_train_E, d_val_E],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(period=Config.VERBOSE_EVAL),
                ],
            )
            self.models_E.append(model_E)

            # Evaluate East
            pred_val_E = model_E.predict(X_val)
            mae_E = np.mean(np.abs(y_val_E - pred_val_E))
            mae_E_scores.append(mae_E)
            print(f"Fold {fold+1} East MAE: {mae_E:.6f}")

            # --- Train North Model ---
            y_train_N, y_val_N = y_N.iloc[train_idx], y_N.iloc[val_idx]
            d_train_N = lgb.Dataset(X_train, label=y_train_N)
            d_val_N = lgb.Dataset(X_val, label=y_val_N, reference=d_train_N)

            model_N = lgb.train(
                self.params,
                d_train_N,
                valid_sets=[d_train_N, d_val_N],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(period=Config.VERBOSE_EVAL),
                ],
            )
            self.models_N.append(model_N)

            # Evaluate North
            pred_val_N = model_N.predict(X_val)
            mae_N = np.mean(np.abs(y_val_N - pred_val_N))
            mae_N_scores.append(mae_N)
            print(f"Fold {fold+1} North MAE: {mae_N:.6f}")

            # Save models to cache
            p_e, p_n = self._get_model_paths(fold)
            model_E.save_model(p_e)
            model_N.save_model(p_n)

        print("\nTraining Complete.")
        print(f"Average East MAE: {np.mean(mae_E_scores):.6f}")
        print(f"Average North MAE: {np.mean(mae_N_scores):.6f}")

    def predict(self, df):
        """
        Predict residuals for the given dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing features.

        Returns:
            tuple: (pred_E, pred_N) numpy arrays.
        """
        X = df[self.feature_cols]
        pred_E = np.zeros(len(X))
        pred_N = np.zeros(len(X))

        if not self.models_E or not self.models_N:
            raise RuntimeError("Models not trained or loaded.")

        # Average predictions from all fold models
        for model in self.models_E:
            pred_E += model.predict(X)
        pred_E /= len(self.models_E)

        for model in self.models_N:
            pred_N += model.predict(X)
        pred_N /= len(self.models_N)

        return pred_E, pred_N


def train_residual_model(train_df, load_cached_model=True):
    """
    Wrapper to initialize and train the regressor.

    Args:
        train_df (pd.DataFrame): Training data.
        load_cached_model (bool): Whether to attempt loading from disk.

    Returns:
        ResidualRegressor: The trained model instance.
    """
    regressor = ResidualRegressor()
    regressor.train(train_df, force_retrain=not load_cached_model)
    return regressor


def apply_correction(df, pred_E, pred_N):
    """
    Apply predicted ENU residuals to WLS positions to get corrected WGS84 coordinates.

    Args:
        df (pd.DataFrame): Dataframe containing WLS ECEF coordinates ('Wls_X', 'Wls_Y', 'Wls_Z').
        pred_E (np.array): Predicted East residual (meters).
        pred_N (np.array): Predicted North residual (meters).

    Returns:
        tuple: (Corrected Latitude, Corrected Longitude)
    """
    # 1. Get WLS Anchor Position (ECEF)
    wls_x = df["Wls_X"].values
    wls_y = df["Wls_Y"].values
    wls_z = df["Wls_Z"].values

    # 2. Convert Anchor to Geodetic (Lat, Lon, Alt)
    # This serves as the reference point for the local ENU frame
    lat0, lon0, alt0 = ECEF_to_WGS84(wls_x, wls_y, wls_z)

    # 3. Convert Predicted ENU Residuals to Global WGS84
    # The predicted residual is the vector from WLS to Truth in the ENU frame.
    # So, Corrected_Pos = Anchor + Residual.
    # We assume Up residual is 0 for horizontal correction.
    pred_U = np.zeros_like(pred_E)

    new_lat, new_lon, _ = ENU_to_WGS84(pred_E, pred_N, pred_U, lat0, lon0, alt0)

    return new_lat, new_lon
