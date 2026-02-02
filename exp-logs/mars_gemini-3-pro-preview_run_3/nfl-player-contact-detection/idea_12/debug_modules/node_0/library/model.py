import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Tuple, Dict, Optional

from library.config import Config
from library.utils import seed_everything, calc_mcc


class DualStreamModel:
    """
    Wrapper class for the Dual-Stream XGBoost architecture.
    Manages training, evaluation, and inference for both Interaction (Stream A)
    and Impact (Stream B) models.
    """

    def __init__(self):
        """
        Initialize the model wrapper. Sets random seeds for reproducibility.
        """
        seed_everything(Config.SEED)
        self.models = {}
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_model_path(self, stream_name: str) -> str:
        """
        Returns the file path for saving/loading the model.
        """
        return os.path.join(self.working_dir, f"{stream_name}_model.json")

    def _undersample(
        self, X: pd.DataFrame, y: np.ndarray, ratio: float
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Performs random undersampling on the majority class (0).

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target vector.
            ratio (float): Ratio of negatives to positives (e.g., 10.0 means 10 negs for 1 pos).

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Undersampled X and y.
        """
        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate number of negatives to keep
        n_keep_neg = int(n_pos * ratio)

        # If we have fewer negatives than requested, keep all
        if n_keep_neg > n_neg:
            n_keep_neg = n_neg

        # Randomly sample negatives
        # Using a fixed seed via numpy (already seeded in __init__)
        sampled_neg_indices = np.random.choice(
            neg_indices, size=n_keep_neg, replace=False
        )

        # Combine indices
        combined_indices = np.concatenate([pos_indices, sampled_neg_indices])
        np.random.shuffle(combined_indices)

        # Subset data
        X_resampled = X.iloc[combined_indices].copy()
        y_resampled = y[combined_indices]

        return X_resampled, y_resampled

    def train_stream(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        stream_config: Dict,
    ):
        """
        Trains an XGBoost model for a specific stream with undersampling and early stopping.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data (used for early stopping).
            stream_config: Configuration dict for the stream (A or B).
        """
        stream_name = stream_config["name"]
        print(f"[{stream_name}] Starting training process...")

        # 1. Undersampling
        ratio = stream_config.get("neg_pos_ratio", 10.0)
        print(f"[{stream_name}] Applying undersampling with ratio {ratio}...")
        X_train_res, y_train_res = self._undersample(X_train, y_train, ratio)

        print(
            f"[{stream_name}] Train shape original: {X_train.shape}, resampled: {X_train_res.shape}"
        )
        print(f"[{stream_name}] Positive samples: {np.sum(y_train_res == 1)}")

        # 2. Initialize Model
        # Copy params to avoid modifying global config
        params = Config.XGB_PARAMS.copy()

        clf = xgb.XGBClassifier(**params)

        # 3. Fit Model
        print(f"[{stream_name}] Fitting XGBoost...")
        clf.fit(
            X_train_res,
            y_train_res,
            eval_set=[(X_val, y_val)],
            verbose=Config.VERBOSE_EVAL,
        )

        # 4. Evaluate
        print(f"[{stream_name}] Evaluating on validation set...")
        # Predict probabilities for threshold optimization later, but here we check default 0.5 for logging
        y_val_pred = clf.predict(X_val)
        mcc = calc_mcc(y_val, y_val_pred)

        # Requirement: print full precision
        print(f"[{stream_name}] Validation MCC: {mcc}")

        # 5. Save Model
        model_path = self._get_model_path(stream_name)
        clf.save_model(model_path)
        print(f"[{stream_name}] Model saved to {model_path}")

        self.models[stream_name] = clf

    def predict_stream(self, X: pd.DataFrame, stream_config: Dict) -> np.ndarray:
        """
        Generates probability predictions for a specific stream.
        Loads the model from disk if not already in memory.

        Args:
            X (pd.DataFrame): Feature matrix.
            stream_config (Dict): Configuration dict for the stream.

        Returns:
            np.ndarray: Predicted probabilities (Class 1).
        """
        stream_name = stream_config["name"]
        model_path = self._get_model_path(stream_name)

        # Load model if not present
        if stream_name not in self.models:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model for {stream_name} not found at {model_path}. Train it first."
                )

            print(f"[{stream_name}] Loading model from {model_path}...")
            clf = xgb.XGBClassifier()
            clf.load_model(model_path)
            self.models[stream_name] = clf
        else:
            clf = self.models[stream_name]

        if X.empty:
            return np.array([])

        # Predict probabilities
        # XGBoost predict_proba returns [prob_0, prob_1]
        probs = clf.predict_proba(X)[:, 1]
        return probs
