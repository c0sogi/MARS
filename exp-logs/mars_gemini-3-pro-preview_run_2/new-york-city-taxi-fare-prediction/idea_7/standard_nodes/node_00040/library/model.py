import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import (
    XGB_PARAMS,
    TRAIN_ROUNDS,
    EARLY_STOPPING_ROUNDS,
    SUBMISSION_PATH,
)


class FarePredictor:
    """
    Encapsulates the XGBoost Regressor for the taxi fare prediction task.
    """

    def __init__(self, params=None):
        """
        Initializes the FarePredictor.

        Args:
            params (dict, optional): Hyperparameters for XGBoost.
                                     Defaults to XGB_PARAMS from library.config.
        """
        self.params = params if params is not None else XGB_PARAMS
        self.model = None

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=None,
        early_stopping_rounds=None,
    ):
        """
        Trains the XGBoost model using the provided training and validation data.

        Args:
            X_train (pd.DataFrame): Features for the training set.
            y_train (np.ndarray): Target values for the training set.
            X_val (pd.DataFrame): Features for the validation set.
            y_val (np.ndarray): Target values for the validation set.
            num_boost_round (int, optional): Maximum number of boosting iterations.
                                             Defaults to TRAIN_ROUNDS from config.
            early_stopping_rounds (int, optional): Rounds without improvement to stop training.
                                                   Defaults to EARLY_STOPPING_ROUNDS from config.
        """
        rounds = num_boost_round if num_boost_round is not None else TRAIN_ROUNDS
        es_rounds = (
            early_stopping_rounds
            if early_stopping_rounds is not None
            else EARLY_STOPPING_ROUNDS
        )

        print(f"Training XGBoost on {len(X_train)} samples...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, "train"), (dval, "val")]

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=rounds,
            evals=evals,
            early_stopping_rounds=es_rounds,
            verbose_eval=50,
        )

        if hasattr(self.model, "best_score"):
            print(f"Best validation score: {self.model.best_score}")

    def predict(self, X_test):
        """
        Generates predictions for the test set.

        Args:
            X_test (pd.DataFrame): Features for the test set.

        Returns:
            np.ndarray: Predicted fare amounts, floored at $2.50.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)

        # Post-processing: Apply minimum fare floor of $2.50
        predictions = np.maximum(predictions, 2.50)

        return predictions

    def save_submission(self, test_keys, predictions, output_path=None):
        """
        Saves the predictions to a CSV file in the required submission format.

        Args:
            test_keys (np.ndarray): Array of 'key' strings corresponding to the test set.
            predictions (np.ndarray): Array of predicted fare amounts.
            output_path (str, optional): File path to save the submission.
                                         Defaults to SUBMISSION_PATH from config.
        """
        if output_path is None:
            output_path = SUBMISSION_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        submission = pd.DataFrame({"key": test_keys, "fare_amount": predictions})

        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
