import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import set_seed, compute_mcc


class XGBWrapper:
    """
    Wrapper class for XGBoost training and inference.
    Handles data preparation (undersampling), training with early stopping,
    model serialization, and prediction.
    """

    def __init__(self, params, model_path):
        """
        Initialize the XGBWrapper.

        Args:
            params (dict): XGBoost hyperparameters.
            model_path (str): Path to save/load the trained model (should end in .json).
        """
        self.params = params
        self.model_path = model_path
        self.model = None

        # Ensure the directory for the model exists
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _undersample(self, X, y):
        """
        Performs random undersampling on the majority class (0).

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.array): Labels.

        Returns:
            tuple: (resampled_X, resampled_y)
        """
        # Ensure inputs are compatible
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        # Calculate number of negatives to keep based on ratio
        n_neg_keep = int(n_pos * Config.UNDERSAMPLE_RATIO)

        # If we have more negatives than we need, sample them
        if n_neg_keep < len(neg_indices):
            # Use fixed seed for reproducibility of the sampling
            np.random.seed(Config.SEED)
            keep_neg_indices = np.random.choice(
                neg_indices, size=n_neg_keep, replace=False
            )
        else:
            keep_neg_indices = neg_indices

        # Combine and shuffle indices
        keep_indices = np.concatenate([pos_indices, keep_neg_indices])
        np.random.shuffle(keep_indices)

        return X.iloc[keep_indices], y[keep_indices]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.array): Training labels.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (np.array, optional): Validation labels.
        """
        set_seed(Config.SEED)

        print(f"Training model. Output path: {self.model_path}")
        print(f"Original Train Shape: {X_train.shape}, Positives: {np.sum(y_train)}")

        # Apply Undersampling to Training Data
        X_train_res, y_train_res = self._undersample(X_train, y_train)
        print(
            f"Resampled Train Shape: {X_train_res.shape}, Positives: {np.sum(y_train_res)}"
        )

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)

        evals = [(dtrain, "train")]
        dval = None

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))

        # Train
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=Config.NUM_BOOST_ROUND,
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        # Save Model
        if self.model_path:
            self.model.save_model(self.model_path)
            print(f"Model saved to {self.model_path}")

        # Validation Metrics
        if dval:
            # Predict probabilities
            y_pred_prob = self.model.predict(dval)

            # Calculate LogLoss
            ll = log_loss(y_val, y_pred_prob)

            # Calculate MCC (using 0.5 threshold for initial check, though optimization happens later)
            y_pred_bin = (y_pred_prob >= 0.5).astype(int)
            mcc = compute_mcc(y_val, y_pred_bin)

            print("Validation Metrics:")
            print(f"LogLoss: {ll}")
            print(f"MCC (thresh=0.5): {mcc}")

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.array: Predicted probabilities.
        """
        # Load model if not in memory
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}")
                self.model = xgb.Booster()
                self.model.load_model(self.model_path)
            else:
                raise RuntimeError("Model not trained and no model file found.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
