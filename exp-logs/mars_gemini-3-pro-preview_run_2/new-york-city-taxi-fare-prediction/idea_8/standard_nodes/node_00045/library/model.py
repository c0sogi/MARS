import os
import numpy as np
import xgboost as xgb
from library.config import (
    XGB_PARAMS,
    MODEL_SAVE_PATH,
    EARLY_STOPPING_ROUNDS,
    MIN_FARE_FLOOR,
    CACHE_DIR,
)


class FarePredictor:
    """
    Wrapper class for the XGBoost Regressor model to handle training,
    evaluation, and prediction for the taxi fare task.
    """

    def __init__(self):
        """
        Initialize the XGBoost Regressor with parameters from config.
        """
        # XGB_PARAMS contains n_estimators, learning_rate, objective, etc.
        self.model = xgb.XGBRegressor(**XGB_PARAMS)

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.
        """
        print(f"Training XGBoost model with params: {XGB_PARAMS}")

        # Fit the model
        # verbose=100 prints every 100 rounds to keep output clean but visible
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )

        # Retrieve and print the best score
        # XGBoost stores evaluation results in evals_result() if needed,
        # but best_score is accessible if early stopping is used.
        # Note: best_score is the score of the best_iteration on the last validation set.

        # Accessing the underlying booster's best score
        best_score = self.model.best_score
        print(f"Best Validation RMSE: {best_score}")

        # Save the model
        self.save_model()

    def save_model(self):
        """
        Saves the trained model to the configured path.
        """
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        # Using the save_model method of the XGBRegressor which calls the booster's save_model
        # JSON format is standard and robust for XGBoost
        self.model.save_model(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    def load_model(self):
        """
        Loads a trained model from the configured path.
        """
        if os.path.exists(MODEL_SAVE_PATH):
            self.model.load_model(MODEL_SAVE_PATH)
            print(f"Model loaded from {MODEL_SAVE_PATH}")
        else:
            print(f"No model found at {MODEL_SAVE_PATH}")

    def predict(self, X_test):
        """
        Generates predictions for the test set.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.array: Predicted fare amounts.
        """
        # Generate raw predictions
        predictions = self.model.predict(X_test)

        # Post-processing: Apply minimum fare floor
        # Taxi fares cannot be arbitrarily low (e.g. negative or near zero)
        predictions = np.maximum(predictions, MIN_FARE_FLOOR)

        return predictions
