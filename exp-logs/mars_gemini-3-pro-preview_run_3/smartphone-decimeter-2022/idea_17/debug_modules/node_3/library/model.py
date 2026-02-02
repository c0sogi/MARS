import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.utils import EnuToGeodetic

# Constants
MODEL_DIR = "./working/idea_17/models"
SEED = 42


class ResidualRegressor:
    """
    Manages an ensemble of LightGBM models to predict ENU position residuals
    based on geometric and signal features.
    """

    def __init__(self, n_folds=5, model_dir=MODEL_DIR):
        self.n_folds = n_folds
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Feature columns based on library.features.py output
        self.feature_cols = [
            "L1_SatCount",
            "L1_TotalWeight",
            "L1_Proj_E",
            "L1_Proj_N",
            "L1_Proj_U",
            "L5_SatCount",
            "L5_TotalWeight",
            "L5_Proj_E",
            "L5_Proj_N",
            "L5_Proj_U",
            "IMU_MeasurementX_mean",
            "IMU_MeasurementX_std",
            "IMU_MeasurementY_mean",
            "IMU_MeasurementY_std",
            "IMU_MeasurementZ_mean",
            "IMU_MeasurementZ_std",
        ]

        # Targets
        self.targets = ["Target_E", "Target_N"]

        # LightGBM Hyperparameters
        self.params = {
            "objective": "mae",  # L1 loss for robustness against outliers
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }

    def _get_model_path(self, target_name, fold):
        return os.path.join(self.model_dir, f"lgbm_{target_name}_fold_{fold}.txt")

    def train(self, train_df, load_cached_models=False):
        """
        Trains the ensemble of LightGBM models.

        Args:
            train_df (pd.DataFrame): Training data with features and targets.
            load_cached_models (bool): If True, skips training if models exist.
        """
        print(f"Training ResidualRegressor with {self.n_folds} folds...")

        # Filter valid data
        train_df = train_df.dropna(subset=self.feature_cols + self.targets).reset_index(
            drop=True
        )

        X = train_df[self.feature_cols]
        groups = train_df["drive_id"]

        gkf = GroupKFold(n_splits=self.n_folds)

        # Store validation scores
        oof_preds = {t: np.zeros(len(train_df)) for t in self.targets}

        for target in self.targets:
            y = train_df[target]
            print(f"\nTarget: {target}")

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
                model_path = self._get_model_path(target, fold)

                # Check cache
                if load_cached_models and os.path.exists(model_path):
                    print(f"  Fold {fold}: Loading cached model...")
                    model = lgb.Booster(model_file=model_path)
                else:
                    print(f"  Fold {fold}: Training...")
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                    model = lgb.LGBMRegressor(**self.params)

                    callbacks = [
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),  # Silence logging
                    ]

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="mae",
                        callbacks=callbacks,
                    )

                    # Save model
                    model.booster_.save_model(model_path)
                    model = model.booster_  # Use booster for consistent prediction API

                # Predict on validation set
                X_val = X.iloc[val_idx]
                y_pred = model.predict(X_val)
                oof_preds[target][val_idx] = y_pred

                # Calculate fold score
                y_val = y.iloc[val_idx]
                mae = np.mean(np.abs(y_val - y_pred))
                print(f"  Fold {fold} MAE: {mae:.8f}")

        # Overall Metrics
        print("\nOverall OOF Metrics:")
        for target in self.targets:
            y_true = train_df[target]
            y_pred = oof_preds[target]
            mae = np.mean(np.abs(y_true - y_pred))
            print(f"  {target} MAE: {mae:.8f}")

    def predict(self, test_df):
        """
        Generates predictions (Anchor Positions) for the test set.

        Args:
            test_df (pd.DataFrame): Test data with features.

        Returns:
            pd.DataFrame: DataFrame with columns ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
                          representing the ML-corrected anchor positions.
        """
        print("\nGenerating Anchor Predictions...")

        # Ensure features exist, fill NaNs with 0 if any (though extraction should handle it)
        X_test = test_df[self.feature_cols].fillna(0)

        pred_residuals = {}

        for target in self.targets:
            preds = np.zeros(len(test_df))

            for fold in range(self.n_folds):
                model_path = self._get_model_path(target, fold)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"Model not found: {model_path}. Train models first."
                    )

                model = lgb.Booster(model_file=model_path)
                preds += model.predict(X_test)

            pred_residuals[target] = preds / self.n_folds

        # Apply residuals to WLS baseline to get Anchor Positions
        # WLS is in Geodetic, Residuals are in ENU (East, North)
        # We assume Up residual is 0 for the anchor generation

        wls_lat = test_df["Wls_Lat"].values
        wls_lon = test_df["Wls_Lon"].values
        wls_alt = test_df["Wls_Alt"].values

        pred_e = pred_residuals["Target_E"]
        pred_n = pred_residuals["Target_N"]
        pred_u = np.zeros_like(pred_e)  # Assume 0 vertical correction

        # Vectorized conversion: ENU (Residuals) + Reference (WLS) -> Geodetic (Anchor)
        # Since utils functions are scalar/numpy-based, we iterate or use numpy vectorization
        # The utils classes are static methods using numpy, so they should vectorize naturally.

        anchor_lat, anchor_lon, anchor_alt = EnuToGeodetic.transform(
            pred_e, pred_n, pred_u, wls_lat, wls_lon, wls_alt
        )

        # Construct result dataframe
        result_df = pd.DataFrame(
            {
                "tripId": test_df["tripId"],
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "LatitudeDegrees": anchor_lat,
                "LongitudeDegrees": anchor_lon,
            }
        )

        return result_df
