import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config


class XGBResiduaLearner:
    """
    Wrapper for XGBoost training and inference, designed for Residual Learning.
    Expects DMatrix objects to have 'base_margin' set, enabling the model
    to learn the residual (Target - Margin).
    """

    def __init__(self, params=None):
        """
        Initialize the learner.

        Args:
            params (dict, optional): XGBoost hyperparameters. Defaults to Config.XGB_PARAMS.
        """
        self.params = params if params is not None else Config.XGB_PARAMS.copy()
        self.model = None

    def train(self, dtrain, dval, num_boost_round=None, early_stopping_rounds=None):
        """
        Trains the XGBoost model.

        Args:
            dtrain (xgb.DMatrix): Training data (features + label + base_margin).
            dval (xgb.DMatrix): Validation data (features + label + base_margin).
            num_boost_round (int, optional): Number of boosting rounds.
            early_stopping_rounds (int, optional): Rounds for early stopping.
        """
        # Set defaults from Config if not provided
        if num_boost_round is None:
            num_boost_round = Config.NUM_BOOST_ROUND
        if early_stopping_rounds is None:
            early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS

        # Watchlist for monitoring
        watchlist = [(dtrain, "train"), (dval, "eval")]

        print(f"Starting training with {num_boost_round} rounds...")

        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        # Print best score with full precision
        if hasattr(self.model, "best_score"):
            print(f"Best validation score: {self.model.best_score}")
            print(f"Best iteration: {self.model.best_iteration}")

    def predict(self, dtest):
        """
        Generates predictions for the test set.

        Args:
            dtest (xgb.DMatrix): Test data (features + base_margin).

        Returns:
            np.array: Final predicted fare amounts (Margin + Residual).
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # Predict
        # Since dtest is constructed with base_margin, the prediction is:
        # y_pred = base_margin + sum(tree_outputs)
        preds = self.model.predict(dtest)

        # Post-processing
        # 1. Apply minimum fare floor ($2.50)
        # 2. Ensure non-negativity (covered by the floor)
        preds = np.maximum(preds, 2.5)

        return preds

    def save(self, filename="xgb_model.json"):
        """
        Saves the trained model to the working directory.
        """
        if self.model is None:
            print("No model to save.")
            return

        save_path = os.path.join(Config.WORKING_DIR, filename)
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")

    def load(self, filepath):
        """
        Loads a trained model from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        self.model = xgb.Booster()
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def generate_submission(self, test_df, predictions, output_path=None):
        """
        Generates the submission CSV file.

        Args:
            test_df (pd.DataFrame): DataFrame containing the 'key' column.
            predictions (np.array): Array of predicted fare amounts.
            output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
        """
        if output_path is None:
            output_path = Config.SUBMISSION_PATH

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
