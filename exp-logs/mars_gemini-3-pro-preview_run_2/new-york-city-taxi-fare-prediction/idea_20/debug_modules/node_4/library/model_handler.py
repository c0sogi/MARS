import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library import config


class TaxiFareRegressor:
    """
    Wrapper for the XGBoost Regressor implementing the Factorized Multi-Moment
    Hierarchical Gradient Boosting strategy.

    Handles training with early stopping, GPU acceleration, and prediction
    with domain-specific post-processing (fare floor).
    """

    def __init__(self, params=None):
        """
        Initialize the regressor.

        Args:
            params (dict, optional): XGBoost hyperparameters.
                                     Defaults to config.XGB_PARAMS.
        """
        # Use a copy to avoid modifying the global config dict
        self.params = params.copy() if params else config.XGB_PARAMS.copy()

        # Extract num_boost_round (n_estimators) from params as it's an argument to train()
        # not a parameter for the booster config itself in the native API.
        self.num_boost_round = self.params.pop("n_estimators", 5000)

        self.model = None
        self.best_score = None
        self.best_iteration = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model using the provided training and validation sets.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
        """
        print(
            f"Initializing XGBoost training on device: {self.params.get('device', 'cpu')}"
        )

        # Create DMatrix objects
        # enable_categorical=False because we have pre-processed/encoded features
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Watchlist for monitoring performance
        watchlist = [(dtrain, "train"), (dval, "eval")]

        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose_eval=config.VERBOSE_EVAL,
        )

        # Store best results
        self.best_iteration = self.model.best_iteration
        self.best_score = self.model.best_score

        print(f"Training finished. Best Iteration: {self.best_iteration}")
        print(f"Best Score (RMSE): {self.best_score}")

        # Validate manually to ensure full precision printing
        preds_val = self.model.predict(dval)
        rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))
        print(f"Final Validation RMSE (Full Precision): {rmse_val}")

    def predict(self, X_test):
        """
        Generates predictions for the test set.
        Applies a minimum fare floor of $2.50.

        Args:
            X_test: Test features.

        Returns:
            np.array: Predicted fare amounts.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(
            dtest, iteration_range=(0, self.best_iteration + 1)
        )

        # Apply domain knowledge: Minimum fare is $2.50
        # This prevents unrealistic low predictions for short trips
        predictions = np.maximum(predictions, 2.50)

        return predictions

    def save_model(self, filename="xgb_model.json"):
        """
        Saves the trained model to the cache directory.

        Args:
            filename: Name of the file.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        path = os.path.join(config.CACHE_DIR, filename)
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load_model(self, filename="xgb_model.json"):
        """
        Loads a trained model from the cache directory.

        Args:
            filename: Name of the file.
        """
        path = os.path.join(config.CACHE_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        self.model = xgb.Booster()
        self.model.load_model(path)
        print(f"Model loaded from {path}")
