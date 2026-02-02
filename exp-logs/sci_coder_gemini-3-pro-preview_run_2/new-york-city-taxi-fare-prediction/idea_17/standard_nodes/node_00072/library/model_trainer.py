import xgboost as xgb
import pandas as pd
import numpy as np
import os
from library.config import XGB_PARAMS, MODEL_FEATURES, SUBMISSION_OUTPUT_PATH


class XGBTrainer:
    """
    Wraps the XGBoost training and inference logic.
    Manages data conversion, model training with early stopping,
    and prediction post-processing.
    """

    def __init__(self):
        self.model = None
        self.features = MODEL_FEATURES
        # Create a copy of params to avoid modifying the global config
        self.params = XGB_PARAMS.copy()

        # Extract training loop arguments that are not booster parameters
        self.num_boost_round = self.params.pop("n_estimators", 5000)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # Map sklearn-style parameters to XGBoost native booster parameters
        if "n_jobs" in self.params:
            self.params["nthread"] = self.params.pop("n_jobs")
        if "random_state" in self.params:
            self.params["seed"] = self.params.pop("random_state")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains the XGBoost model using the provided training and validation DataFrames.

        Args:
            train_df: Processed training data.
            val_df: Processed validation data.
        """
        print("Converting training data to DMatrix...")
        dtrain = xgb.DMatrix(train_df[self.features], label=train_df["fare_amount"])

        print("Converting validation data to DMatrix...")
        dval = xgb.DMatrix(val_df[self.features], label=val_df["fare_amount"])

        # Watchlist for monitoring performance
        evals = [(dtrain, "train"), (dval, "eval")]

        print(f"Starting training with {self.num_boost_round} rounds...")
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,  # Print metrics every 100 rounds
        )

        # Print the best validation score in full precision
        print(f"Training finished. Best validation RMSE: {self.model.best_score}")

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the test set and applies post-processing.

        Args:
            test_df: Processed test data.

        Returns:
            Numpy array of predicted fare amounts.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        dtest = xgb.DMatrix(test_df[self.features])
        predictions = self.model.predict(dtest)

        # Post-processing: Apply minimum fare floor ($2.50)
        # This ensures no predictions are below the base fare
        predictions = np.maximum(predictions, 2.50)

        return predictions

    def generate_submission(self, test_df: pd.DataFrame, predictions: np.ndarray):
        """
        Saves the predictions to the submission CSV file in the required format.

        Args:
            test_df: Original test DataFrame containing the 'key' column.
            predictions: Array of predicted fare amounts.
        """
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)

        # Save to CSV
        submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_OUTPUT_PATH}")
