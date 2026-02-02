import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config


class EnergyPredictor:
    """
    A wrapper class for XGBoost regressors to handle multi-output regression
    for formation energy and bandgap energy prediction.

    It trains a separate XGBoost model for each target variable to allow for
    target-specific optimization and early stopping.
    """

    def __init__(self):
        """
        Initialize the EnergyPredictor with parameters from Config.
        """
        self.models = {}
        self.targets = Config.TARGET_COLS
        self.params = Config.XGB_PARAMS.copy()

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains an XGBoost model for each target variable.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Validation targets.
        """
        print(f"Training EnergyPredictor on {len(X_train)} samples...")

        for target in self.targets:
            print(f"\n--- Training model for target: {target} ---")

            # Instantiate the regressor
            model = xgb.XGBRegressor(**self.params)

            # Prepare evaluation sets
            # XGBoost expects eval_set to be a list of (X, y) tuples
            eval_set = [(X_train, y_train[target]), (X_val, y_val[target])]

            # Train the model
            model.fit(
                X_train,
                y_train[target],
                eval_set=eval_set,
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                verbose=Config.VERBOSE_EVAL,
            )

            # Save the trained model
            self.models[target] = model

            # Evaluate on validation set
            preds_val = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val[target], preds_val))

            # Print metric with full precision as requested
            print(f"Validation RMSE for {target}: {rmse}")

    def predict(self, X):
        """
        Generates predictions for the input features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Predictions for each target variable.
        """
        predictions = {}

        for target in self.targets:
            if target in self.models:
                # Predict using the specific model for this target
                predictions[target] = self.models[target].predict(X)
            else:
                raise ValueError(f"Model for target '{target}' has not been trained.")

        return pd.DataFrame(predictions, index=X.index)
