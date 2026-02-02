import lightgbm as lgb
import pandas as pd
import numpy as np
from library.config import Config


class ResidualRegressor:
    def __init__(self):
        self.model_lat = None
        self.model_lon = None
        self.feature_cols = None
        # Columns that are not input features for the model
        self.meta_cols = ["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]

    def _get_features(self, df):
        """
        Extracts feature columns from the dataframe, excluding metadata.
        """
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c not in self.meta_cols]
        return df[self.feature_cols]

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains two LightGBM models: one for latitude residual, one for longitude residual.

        Args:
            X_train (pd.DataFrame): Training features (including metadata).
            y_train (pd.DataFrame): Training targets (residuals).
            X_val (pd.DataFrame): Validation features (including metadata).
            y_val (pd.DataFrame): Validation targets (residuals).
        """
        # Prepare features
        X_train_feats = self._get_features(X_train)
        X_val_feats = self._get_features(X_val)

        print(f"Training with {len(self.feature_cols)} features.")

        # Prepare parameters
        params = Config.LGBM_PARAMS.copy()
        # n_estimators is passed as num_boost_round to lgb.train
        num_boost_round = params.pop("n_estimators", 5000)

        # --- Train Latitude Model ---
        print("\n--- Training Latitude Model ---")
        dtrain_lat = lgb.Dataset(X_train_feats, label=y_train["target_lat"])
        dval_lat = lgb.Dataset(
            X_val_feats, label=y_val["target_lat"], reference=dtrain_lat
        )

        self.model_lat = lgb.train(
            params,
            dtrain_lat,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain_lat, dval_lat],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=Config.VERBOSE_EVAL),
            ],
        )

        # --- Train Longitude Model ---
        print("\n--- Training Longitude Model ---")
        dtrain_lon = lgb.Dataset(X_train_feats, label=y_train["target_lon"])
        dval_lon = lgb.Dataset(
            X_val_feats, label=y_val["target_lon"], reference=dtrain_lon
        )

        self.model_lon = lgb.train(
            params,
            dtrain_lon,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain_lon, dval_lon],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=Config.VERBOSE_EVAL),
            ],
        )

    def predict(self, X):
        """
        Predicts latitude and longitude residuals.

        Args:
            X (pd.DataFrame): Features dataframe (including metadata).

        Returns:
            pd.DataFrame: DataFrame with columns ['pred_lat_res', 'pred_lon_res'].
        """
        X_feats = self._get_features(X)

        if self.model_lat is None or self.model_lon is None:
            raise RuntimeError("Models have not been trained yet. Call fit() first.")

        pred_lat = self.model_lat.predict(
            X_feats, num_iteration=self.model_lat.best_iteration
        )
        pred_lon = self.model_lon.predict(
            X_feats, num_iteration=self.model_lon.best_iteration
        )

        return pd.DataFrame({"pred_lat_res": pred_lat, "pred_lon_res": pred_lon})
