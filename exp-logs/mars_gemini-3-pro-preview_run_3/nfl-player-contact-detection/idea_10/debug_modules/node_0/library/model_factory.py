import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, log_loss
from typing import Dict, Optional, Tuple, Union

from library.config import Config
from library.utils import seed_everything


class ContactGBDT:
    """
    Wrapper class for XGBoost training and inference, specifically tailored
    for the Contact Detection task. Handles Stream A (Interaction) and
    Stream B (Impact) configurations.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the GBDT model with specific parameters.

        Args:
            params (Dict): XGBoost hyperparameters. If None, uses defaults.
        """
        self.params = params if params is not None else {}
        self.model = None

        # Set seed for reproducibility
        seed_everything(Config.SEED)

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Trains the XGBoost model with Early Stopping.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (used for early stopping).
            y_val: Validation labels.
            verbose: Whether to print training progress.
        """
        # Initialize the XGBClassifier
        # We filter out params that are not arguments to the constructor if necessary,
        # but passing **self.params usually works if keys are valid.
        # Note: 'early_stopping_rounds' is passed to fit(), not __init__ in newer sklearn APIs,
        # but we can extract it from params if present.

        fit_params = self.params.copy()
        early_stopping_rounds = fit_params.pop("early_stopping_rounds", 50)

        self.model = xgb.XGBClassifier(**fit_params)

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        if verbose:
            print(f"Starting training with params: {fit_params}")
            print(f"Training data shape: {X_train.shape}")
            if X_val is not None:
                print(f"Validation data shape: {X_val.shape}")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=100 if verbose else False,
        )

        if verbose and X_val is not None and y_val is not None:
            self._evaluate(X_val, y_val)

    def _evaluate(self, X_val, y_val):
        """
        Internal method to print validation metrics with full precision.
        """
        # Predict probabilities
        y_pred_prob = self.model.predict_proba(X_val)[:, 1]

        # Calculate Log Loss
        loss = log_loss(y_val, y_pred_prob)

        # Calculate MCC at default 0.5 threshold (just for reference)
        y_pred_bin = (y_pred_prob > 0.5).astype(int)
        mcc = matthews_corrcoef(y_val, y_pred_bin)

        print("--- Validation Metrics ---")
        print(f"Log Loss: {loss}")
        print(f"MCC (thresh=0.5): {mcc}")
        print("--------------------------")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features.

        Returns:
            np.ndarray: Probabilities for the positive class (contact).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        """
        Save the trained model to a JSON file.

        Args:
            path: File path.
        """
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load a model from a JSON file.

        Args:
            path: File path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        # We need to initialize the model structure before loading weights
        # We use the params stored in self.params
        fit_params = self.params.copy()
        # Remove fit-specific params if they exist in the dict
        fit_params.pop("early_stopping_rounds", None)

        self.model = xgb.XGBClassifier(**fit_params)
        self.model.load_model(path)
        print(f"Model loaded from {path}")
