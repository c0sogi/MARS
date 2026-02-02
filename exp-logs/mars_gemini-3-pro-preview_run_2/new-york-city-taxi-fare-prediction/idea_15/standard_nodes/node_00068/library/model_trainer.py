import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config


class XGBTrainer:
    """
    Wrapper for XGBoost training and inference.
    Utilizes parameters defined in Config to train a regressor on the Learner set.
    """

    def __init__(self):
        # Make a copy to avoid modifying the global config
        self.params = Config.XGB_PARAMS.copy()

        # Extract training control parameters that are passed to xgb.train directly
        self.num_boost_round = self.params.pop("n_estimators", 10000)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model using the provided training and validation data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.

        Returns:
            float: The RMSE score on the validation set.
        """
        print(f"Initializing DMatrix for training (Train shape: {X_train.shape})...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        watchlist = [(dtrain, "train"), (dval, "validation")]

        print("Starting XGBoost training...")
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=50,
        )

        # Save the model artifact
        os.makedirs(os.path.dirname(Config.MODEL_OUTPUT_PATH), exist_ok=True)
        self.model.save_model(Config.MODEL_OUTPUT_PATH)
        print(f"Model saved to {Config.MODEL_OUTPUT_PATH}")

        # Calculate and print final validation metric
        # We use the best iteration if early stopping occurred
        preds_val = self.model.predict(
            dval, iteration_range=(0, self.model.best_iteration + 1)
        )
        rmse = np.sqrt(mean_squared_error(y_val, preds_val))

        print(f"Final Validation RMSE: {rmse}")
        return rmse

    def predict(self, X_test):
        """
        Generates predictions for the test set.
        Applies post-processing (minimum fare floor).

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted fare amounts.
        """
        if self.model is None:
            if os.path.exists(Config.MODEL_OUTPUT_PATH):
                print(f"Loading model from {Config.MODEL_OUTPUT_PATH}...")
                self.model = xgb.Booster()
                self.model.load_model(Config.MODEL_OUTPUT_PATH)
            else:
                raise ValueError("Model has not been trained and no artifact found.")

        print(f"Generating predictions for {len(X_test)} samples...")
        dtest = xgb.DMatrix(X_test)

        # Predict using the best iteration
        try:
            best_iteration = self.model.best_iteration
            preds = self.model.predict(dtest, iteration_range=(0, best_iteration + 1))
        except AttributeError:
            # Fallback if best_iteration is not set (e.g. loaded model without training context)
            preds = self.model.predict(dtest)

        # Post-processing: Apply minimum fare floor ($2.50)
        # Taxi fares in NYC start at $2.50
        preds = np.maximum(preds, 2.50)

        return preds


def train_and_predict(X_train, y_train, X_val, y_val, X_test, test_keys):
    """
    Orchestrates the training, prediction, and submission generation process.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test: Test features.
        test_keys: Series or array of keys corresponding to X_test.
    """
    trainer = XGBTrainer()

    # Train
    trainer.train(X_train, y_train, X_val, y_val)

    # Predict
    predictions = trainer.predict(X_test)

    # Generate Submission
    print("Generating submission file...")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission = pd.DataFrame({"key": test_keys, "fare_amount": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return predictions
