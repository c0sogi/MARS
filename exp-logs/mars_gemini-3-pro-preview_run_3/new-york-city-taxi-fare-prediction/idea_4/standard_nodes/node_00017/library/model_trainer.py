import os
import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import Config


class ModelManager:
    """
    Manages the training, evaluation, and inference of the Heterogeneous Ensemble
    (XGBoost + LightGBM) for the Taxi Fare Prediction task.
    """

    def __init__(self):
        """
        Initialize the ModelManager with paths for model persistence.
        """
        self.working_dir = Config.WORKING_DIR
        self.xgb_path = os.path.join(self.working_dir, "xgboost_model.joblib")
        self.lgbm_path = os.path.join(self.working_dir, "lgbm_model.joblib")

        # In-memory model storage
        self.xgb_model = None
        self.lgbm_model = None

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost Regressor using GPU acceleration.

        Args:
            X_train, y_train: Training features and target.
            X_val, y_val: Validation features and target for early stopping.

        Returns:
            Trained XGBRegressor model.
        """
        print("Initializing XGBoost Regressor...")

        # Prepare parameters
        params = Config.XGB_PARAMS.copy()

        # Initialize model
        # early_stopping_rounds is passed via params to the constructor in newer XGBoost versions
        model = xgb.XGBRegressor(**params)

        print(f"Training XGBoost on {len(X_train)} samples...")

        # Train with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        print("Evaluating XGBoost...")
        val_preds = model.predict(X_val)
        # Calculate RMSE using numpy sqrt + MSE for version compatibility
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"XGBoost Validation RMSE: {rmse}")

        # Save model
        print(f"Saving XGBoost model to {self.xgb_path}...")
        joblib.dump(model, self.xgb_path)
        self.xgb_model = model

        return model

    def train_lgbm(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM Regressor using CPU optimization.

        Args:
            X_train, y_train: Training features and target.
            X_val, y_val: Validation features and target for early stopping.

        Returns:
            Trained LGBMRegressor model.
        """
        print("Initializing LightGBM Regressor...")

        # Prepare parameters
        params = Config.LGBM_PARAMS.copy()
        early_stopping_rounds = params.pop("early_stopping_rounds", 100)

        # Initialize model
        model = lgb.LGBMRegressor(**params)

        print(f"Training LightGBM on {len(X_train)} samples...")

        # Configure callbacks for early stopping and logging suppression
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        # Train
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # Evaluate
        print("Evaluating LightGBM...")
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"LightGBM Validation RMSE: {rmse}")

        # Save model
        print(f"Saving LightGBM model to {self.lgbm_path}...")
        joblib.dump(model, self.lgbm_path)
        self.lgbm_model = model

        return model

    def predict(self, X_test):
        """
        Generates final ensemble predictions for the test set.
        Loads models from disk if they are not currently in memory.

        Args:
            X_test: Test features.

        Returns:
            numpy array of predicted fare amounts.
        """
        # Ensure XGBoost model is loaded
        if self.xgb_model is None:
            if os.path.exists(self.xgb_path):
                print(f"Loading XGBoost model from {self.xgb_path}...")
                self.xgb_model = joblib.load(self.xgb_path)
            else:
                raise FileNotFoundError(
                    f"XGBoost model not found at {self.xgb_path}. Train it first."
                )

        # Ensure LightGBM model is loaded
        if self.lgbm_model is None:
            if os.path.exists(self.lgbm_path):
                print(f"Loading LightGBM model from {self.lgbm_path}...")
                self.lgbm_model = joblib.load(self.lgbm_path)
            else:
                raise FileNotFoundError(
                    f"LightGBM model not found at {self.lgbm_path}. Train it first."
                )

        print("Generating ensemble predictions...")

        # Generate individual predictions
        pred_xgb = self.xgb_model.predict(X_test)
        pred_lgbm = self.lgbm_model.predict(X_test)

        # Weighted Average Ensemble
        w_xgb = Config.WEIGHT_XGB
        w_lgbm = Config.WEIGHT_LGBM

        # Normalize weights just in case, though Config should sum to 1.0 ideally
        total_weight = w_xgb + w_lgbm
        final_preds = (pred_xgb * w_xgb + pred_lgbm * w_lgbm) / total_weight

        return final_preds
