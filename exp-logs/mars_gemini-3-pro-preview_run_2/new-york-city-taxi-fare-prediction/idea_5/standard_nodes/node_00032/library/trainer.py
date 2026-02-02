import os
import xgboost as xgb
import numpy as np
import pandas as pd

from library.config import Config
from library.dataset import TaxiDataset


class Trainer:
    """
    Trainer class for the XGBoost model.
    Handles training, validation, and submission generation.
    """

    def __init__(self):
        # Cite solution_lesson_node_00030: Using XGBoost for tabular spatial data.
        self.model = xgb.XGBRegressor(**Config.XGB_PARAMS)
        self.best_rmse = float("inf")

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with early stopping.
        """
        print(f"Training XGBoost with {X_train.shape[0]} samples...")

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True,
        )

        self.best_rmse = self.model.best_score
        print(f"Training complete. Best RMSE: {self.best_rmse}")

        # Save model
        self.model.save_model(Config.MODEL_SAVE_PATH)

    def validate(self, X_val, y_val):
        """
        Evaluates the model on the validation set using RMSE.
        """
        preds = self.model.predict(X_val)

        # Apply post-processing floor
        preds = np.maximum(preds, Config.MIN_FARE_PREDICTION)

        mse = np.mean((preds - y_val) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def generate_submission(self):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        # Load Test Data
        test_dataset = TaxiDataset(split="test", load_cached_data=True)
        X_test, _, keys = test_dataset.get_data()

        # Load Best Model
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Cannot generate submission.")
            return

        self.model.load_model(Config.MODEL_SAVE_PATH)

        preds = self.model.predict(X_test)

        # Apply Post-Processing (Floor)
        preds = np.maximum(preds, Config.MIN_FARE_PREDICTION)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"key": keys, "fare_amount": preds})

        # Save to CSV
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
