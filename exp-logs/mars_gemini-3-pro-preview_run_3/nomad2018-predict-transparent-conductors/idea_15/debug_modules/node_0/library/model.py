import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.data import TargetTransformer


class DualTargetRegressor:
    """
    Wrapper for training independent XGBoost regressors for formation energy and bandgap energy.
    Handles log-transformation of targets, feature selection, and prediction.
    """

    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.target_transformer = TargetTransformer()
        self.targets = Config.TARGET_COLS
        self.params = Config.XGB_MODEL_PARAMS

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata and target columns.
        """
        exclude = set(self.targets + ["id", "file_path"])
        return [c for c in df.columns if c not in exclude]

    def _remove_constant_features(self, X_train):
        """
        Identifies and returns a list of non-constant feature names.
        """
        # Drop columns where all values are the same
        return [col for col in X_train.columns if X_train[col].nunique() > 1]

    def fit(self, train_df, val_df):
        """
        Trains the XGBoost models.

        Args:
            train_df (pd.DataFrame): Training data with features and targets.
            val_df (pd.DataFrame): Validation data with features and targets.
        """
        # 1. Identify initial feature set
        initial_features = self._get_feature_columns(train_df)

        X_train_full = train_df[initial_features]
        X_val_full = val_df[initial_features]

        # 2. Remove constant features based on training set
        self.feature_cols = self._remove_constant_features(X_train_full)
        print(
            f"Selected {len(self.feature_cols)} features after removing constant columns."
        )

        X_train = X_train_full[self.feature_cols]
        X_val = X_val_full[self.feature_cols]

        # 3. Train a model for each target
        for target in self.targets:
            print(f"\n--- Training for Target: {target} ---")

            # Prepare targets (Log Transform)
            y_train = train_df[target].values
            y_val = val_df[target].values

            y_train_log = self.target_transformer.transform(y_train)
            y_val_log = self.target_transformer.transform(y_val)

            # Initialize XGBoost Regressor
            model = xgb.XGBRegressor(**self.params)

            # Fit with early stopping
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                verbose=Config.VERBOSE_EVAL,
            )

            self.models[target] = model

            # Evaluation
            best_iteration = model.best_iteration
            print(f"Best iteration: {best_iteration}")

            # Predict on validation set
            preds_log = model.predict(X_val)
            preds_original = self.target_transformer.inverse_transform(preds_log)

            # Calculate metrics
            mse_log = mean_squared_error(y_val_log, preds_log)
            rmse_log = np.sqrt(mse_log)

            # RMSLE on original scale is equivalent to RMSE on log scale if log1p is used
            # Metric for competition is Column-wise root mean squared logarithmic error
            # Since we predict log(1+y), RMSE of our prediction IS the RMSLE.
            print(f"Validation RMSLE (Log-Scale RMSE): {rmse_log}")

            # Also print RMSE on original scale for physical intuition
            mse_orig = mean_squared_error(y_val, preds_original)
            rmse_orig = np.sqrt(mse_orig)
            print(f"Validation RMSE (Original Scale): {rmse_orig}")

    def predict(self, test_df):
        """
        Generates predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: DataFrame with 'id' and predicted targets.
        """
        if not self.models:
            raise RuntimeError("Models have not been trained. Call fit() first.")

        # Prepare features
        X_test = test_df[self.feature_cols]

        results = pd.DataFrame()
        results["id"] = test_df["id"]

        for target in self.targets:
            model = self.models[target]

            # Predict (Log scale)
            preds_log = model.predict(X_test)

            # Inverse Transform
            preds_original = self.target_transformer.inverse_transform(preds_log)

            results[target] = preds_original

        return results
