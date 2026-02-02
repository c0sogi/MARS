import os
import xgboost as xgb
import numpy as np
import pandas as pd
from library.config import Config


class Trainer:
    """
    Trainer class for the XGBoost model.
    Handles training, validation, and submission generation.
    """

    def __init__(self):
        # Cite solution_lesson_node_00030: Switching to XGBoost for better spatial partitioning
        self.model = xgb.XGBRegressor(
            n_estimators=Config.N_ESTIMATORS,
            max_depth=Config.MAX_DEPTH,
            learning_rate=Config.LEARNING_RATE,
            subsample=Config.SUBSAMPLE,
            colsample_bytree=Config.COLSAMPLE_BYTREE,
            tree_method=Config.TREE_METHOD,
            device=Config.DEVICE,
            objective=Config.OBJECTIVE,
            n_jobs=-1,
            random_state=Config.SEED,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with early stopping.
        """
        print("Starting XGBoost training...")
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        print("Training complete.")

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

    def generate_submission(self, X_test, keys):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        preds = self.model.predict(X_test)
        preds = np.maximum(preds, Config.MIN_FARE_PREDICTION)

        submission_df = pd.DataFrame({"key": keys, "fare_amount": preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
