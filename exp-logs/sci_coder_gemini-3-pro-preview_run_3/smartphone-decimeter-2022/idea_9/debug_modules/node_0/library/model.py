import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config


class LGBMResidualModel:
    """
    Manages the training and inference of LightGBM models for estimating
    position residuals (East/North error) based on sensor features.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.model_east_path = os.path.join(self.working_dir, "lgbm_east.txt")
        self.model_north_path = os.path.join(self.working_dir, "lgbm_north.txt")
        self.model_east = None
        self.model_north = None
        self.feature_cols = []

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata and targets.
        """
        exclude_cols = [
            "tripId",
            "UnixTimeMillis",
            "drive_id",
            "phone_name",
            "WlsLat",
            "WlsLon",
            "WlsAlt",
            "target_east",
            "target_north",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "LatitudeDegrees_gt",
            "LongitudeDegrees_gt",
        ]
        features = [c for c in df.columns if c not in exclude_cols]
        return features

    def train(self, train_df, val_df):
        """
        Trains two LightGBM models: one for Easting error, one for Northing error.

        Args:
            train_df (pd.DataFrame): Training data with features and targets.
            val_df (pd.DataFrame): Validation data for early stopping.
        """
        self.feature_cols = self._get_feature_columns(train_df)
        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

        # Prepare Datasets
        X_train = train_df[self.feature_cols]
        y_train_east = train_df["target_east"]
        y_train_north = train_df["target_north"]

        X_val = val_df[self.feature_cols]
        y_val_east = val_df["target_east"]
        y_val_north = val_df["target_north"]

        # --- Train East Model ---
        print("\n--- Training East Model ---")
        dtrain_east = lgb.Dataset(X_train, label=y_train_east)
        dval_east = lgb.Dataset(X_val, label=y_val_east, reference=dtrain_east)

        self.model_east = lgb.train(
            Config.LGBM_PARAMS,
            dtrain_east,
            num_boost_round=Config.LGBM_PARAMS["n_estimators"],
            valid_sets=[dtrain_east, dval_east],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=Config.VERBOSE_EVAL),
            ],
        )

        # Save East Model
        self.model_east.save_model(self.model_east_path)
        print(f"East Model saved to {self.model_east_path}")

        # --- Train North Model ---
        print("\n--- Training North Model ---")
        dtrain_north = lgb.Dataset(X_train, label=y_train_north)
        dval_north = lgb.Dataset(X_val, label=y_val_north, reference=dtrain_north)

        self.model_north = lgb.train(
            Config.LGBM_PARAMS,
            dtrain_north,
            num_boost_round=Config.LGBM_PARAMS["n_estimators"],
            valid_sets=[dtrain_north, dval_north],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=Config.VERBOSE_EVAL),
            ],
        )

        # Save North Model
        self.model_north.save_model(self.model_north_path)
        print(f"North Model saved to {self.model_north_path}")

        # --- Validation Metrics ---
        val_pred_east = self.model_east.predict(
            X_val, num_iteration=self.model_east.best_iteration
        )
        val_pred_north = self.model_north.predict(
            X_val, num_iteration=self.model_north.best_iteration
        )

        mae_east = np.mean(np.abs(y_val_east - val_pred_east))
        mae_north = np.mean(np.abs(y_val_north - val_pred_north))

        print("\n=== Final Validation Metrics ===")
        print(f"East MAE: {mae_east}")
        print(f"North MAE: {mae_north}")
        print(f"Average MAE: {(mae_east + mae_north) / 2}")

    def predict(self, test_df):
        """
        Generates residual predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test data with features.

        Returns:
            pd.DataFrame: DataFrame containing ['tripId', 'UnixTimeMillis', 'pred_east', 'pred_north']
        """
        # Load models if not in memory
        if self.model_east is None:
            if os.path.exists(self.model_east_path):
                self.model_east = lgb.Booster(model_file=self.model_east_path)
            else:
                raise FileNotFoundError("East model not found. Train first.")

        if self.model_north is None:
            if os.path.exists(self.model_north_path):
                self.model_north = lgb.Booster(model_file=self.model_north_path)
            else:
                raise FileNotFoundError("North model not found. Train first.")

        # Identify features
        # If feature_cols is empty (e.g. loaded from disk without training), infer from df
        if not self.feature_cols:
            self.feature_cols = self._get_feature_columns(test_df)
            # Ensure model feature names match
            model_features = self.model_east.feature_name()
            # Check if all model features are present
            missing = [f for f in model_features if f not in self.feature_cols]
            if missing:
                raise ValueError(
                    f"Test data missing features required by model: {missing}"
                )
            # Use the order expected by the model
            self.feature_cols = model_features

        X_test = test_df[self.feature_cols]

        print(f"Predicting for {len(X_test)} samples...")
        pred_east = self.model_east.predict(
            X_test, num_iteration=self.model_east.best_iteration
        )
        pred_north = self.model_north.predict(
            X_test, num_iteration=self.model_north.best_iteration
        )

        result = test_df[["tripId", "UnixTimeMillis"]].copy()
        result["pred_east"] = pred_east
        result["pred_north"] = pred_north

        return result
