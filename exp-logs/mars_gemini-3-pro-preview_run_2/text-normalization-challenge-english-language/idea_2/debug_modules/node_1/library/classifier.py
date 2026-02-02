import os
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import set_seed


class TokenClassifier:
    """
    Wrapper for XGBoost Classifier tailored for the Text Normalization task.
    Handles training, evaluation, saving, and loading of the model.
    """

    def __init__(self):
        """
        Initialize the classifier with parameters from Config.
        """
        set_seed(Config.SEED)
        # Make a copy to avoid modifying the global config
        self.params = Config.XGB_PARAMS.copy()
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model using the provided training and validation data.
        Implements early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (np.ndarray): Validation labels.
        """
        print("TokenClassifier: Converting data to DMatrix format...")
        # DMatrix is optimized for XGBoost memory and speed
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Extract training control parameters from the params dict
        # These are usually passed as arguments to train(), not inside the params dict
        num_boost_round = self.params.pop("n_estimators", 1000)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # Watchlist for monitoring performance
        watchlist = [(dtrain, "train"), (dval, "val")]

        print(f"TokenClassifier: Starting training on {Config.XGB_PARAMS['device']}...")
        print(f"  - Max Rounds: {num_boost_round}")
        print(f"  - Early Stopping: {early_stopping_rounds}")

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,  # Print progress every 50 rounds
        )

        print("TokenClassifier: Training complete.")

        # Evaluate on validation set to print full precision metric
        print("TokenClassifier: Evaluating on validation set...")
        # multi:softmax returns class labels directly
        preds = self.model.predict(dval)

        # Ensure predictions are integers (XGBoost might return floats)
        preds = preds.astype(int)

        acc = accuracy_score(y_val, preds)
        print(f"Validation Accuracy: {acc}")

    def predict(self, X):
        """
        Generates predictions for the given features.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Predicted class indices.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return preds.astype(int)

    def save(self, path=None):
        """
        Saves the trained model to disk.

        Args:
            path (str, optional): Path to save the model. Defaults to Config.MODEL_FILE.
        """
        if self.model is None:
            print("Warning: No model to save.")
            return

        target_path = path if path else Config.MODEL_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        self.model.save_model(target_path)
        print(f"TokenClassifier: Model saved to {target_path}")

    def load(self, path=None):
        """
        Loads a trained model from disk.

        Args:
            path (str, optional): Path to load the model from. Defaults to Config.MODEL_FILE.
        """
        target_path = path if path else Config.MODEL_FILE

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Model file not found at {target_path}")

        self.model = xgb.Booster()
        self.model.load_model(target_path)
        print(f"TokenClassifier: Model loaded from {target_path}")
