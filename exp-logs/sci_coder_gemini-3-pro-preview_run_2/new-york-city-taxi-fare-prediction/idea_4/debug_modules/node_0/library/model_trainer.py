import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config
from library.utils import calculate_rmse


class XGBoostTrainer:
    """
    Wrapper class for XGBoost Regressor to handle training, evaluation, and prediction
    specific to the NYC Taxi Fare Prediction task.
    """

    def __init__(self, params=None, n_estimators=None):
        """
        Initialize the XGBoost Trainer.

        Args:
            params (dict, optional): Hyperparameters for XGBoost. If None, uses Config.XGB_PARAMS.
            n_estimators (int, optional): Number of boosting rounds. If None, uses Config.NUM_BOOST_ROUND.
        """
        self.params = params.copy() if params else Config.XGB_PARAMS.copy()

        # Set n_estimators (boosting rounds)
        if n_estimators is not None:
            self.params["n_estimators"] = n_estimators
        elif "n_estimators" not in self.params:
            self.params["n_estimators"] = Config.NUM_BOOST_ROUND

        # Initialize the Regressor
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names = None

    def train(self, train_df, val_df, target_col="fare_amount", key_col="key"):
        """
        Trains the XGBoost model with early stopping.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            target_col (str): Name of the target column.
            key_col (str): Name of the ID column to exclude from features.

        Returns:
            xgb.XGBRegressor: The trained model.
        """
        # Identify feature columns: all columns except key, target, and raw datetime
        # Note: pickup_datetime is usually dropped in feature engineering, but we check just in case.
        exclude_cols = {key_col, target_col, "pickup_datetime"}
        self.feature_names = [c for c in train_df.columns if c not in exclude_cols]

        print(f"Training with {len(self.feature_names)} features: {self.feature_names}")

        X_train = train_df[self.feature_names]
        y_train = train_df[target_col]

        X_val = val_df[self.feature_names]
        y_val = val_df[target_col]

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=Config.VERBOSE_EVAL,
        )

        # Calculate and print final validation metric
        # We predict manually to ensure we verify the final state of the model
        val_preds = self.model.predict(X_val)
        val_rmse = calculate_rmse(y_val, val_preds)

        # Printing full precision as requested
        print(f"Final Validation RMSE (Log Scale): {val_rmse}")

        return self.model

    def predict(self, test_df, key_col="key"):
        """
        Generate predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test data.
            key_col (str): Name of the ID column to exclude.

        Returns:
            np.ndarray: Predicted values (log-scale).
        """
        if self.feature_names is None:
            # Fallback if predicting without training in this session (e.g. after loading)
            exclude_cols = {key_col, "pickup_datetime"}
            feature_cols = [c for c in test_df.columns if c not in exclude_cols]
        else:
            feature_cols = self.feature_names

        X_test = test_df[feature_cols]
        return self.model.predict(X_test)

    def save_model(self, filename="xgb_model.json"):
        """
        Save the trained model to the working directory.
        """
        save_path = os.path.join(Config.WORKING_DIR, filename)
        # Ensure directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, filename="xgb_model.json"):
        """
        Load a trained model from the working directory.
        """
        load_path = os.path.join(Config.WORKING_DIR, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        self.model.load_model(load_path)
        print(f"Model loaded from {load_path}")
