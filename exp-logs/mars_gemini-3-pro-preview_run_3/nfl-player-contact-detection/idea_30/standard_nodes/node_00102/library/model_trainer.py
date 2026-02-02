import xgboost as xgb
import pandas as pd
import numpy as np
import os
import joblib
from library.config import ProjectConfig
from library.utils import get_logger

logger = get_logger("ModelTrainer")


class DualStreamTrainer:
    """
    Manages the training and inference of the Physically-Consistent Hybrid-Context
    Dual-Stream GBDT models.

    Attributes:
        model_a (xgb.Booster): Trained model for Stream A (Player-Player).
        model_b (xgb.Booster): Trained model for Stream B (Player-Ground).
    """

    def __init__(self):
        self.config = ProjectConfig
        self.model_a = None
        self.model_b = None
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def train_stream(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        stream_type: str,
    ) -> xgb.Booster:
        """
        Trains a single XGBoost stream (A or B) using the specific configuration
        for that stream.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            stream_type (str): 'A' for Interaction Stream, 'B' for Impact Stream.

        Returns:
            xgb.Booster: The trained XGBoost model.
        """
        logger.info(f"Initializing training for Stream {stream_type}...")

        # Select parameters based on stream type
        if stream_type == "A":
            params = self.config.XGB_PARAMS_STREAM_A.copy()
        elif stream_type == "B":
            params = self.config.XGB_PARAMS_STREAM_B.copy()
        else:
            raise ValueError(f"Invalid stream_type: {stream_type}. Must be 'A' or 'B'.")

        # Prepare DMatrices
        # nthread is set in params via 'n_jobs', but DMatrix can also take it.
        # We rely on params.
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Watchlist for evaluation
        watchlist = [(dtrain, "train"), (dval, "validation")]

        evals_result = {}

        logger.info(
            f"Starting XGBoost training for Stream {stream_type} with params: {params}"
        )

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=params["n_estimators"],
            evals=watchlist,
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=self.config.VERBOSE_EVAL,
        )

        # Extract best score (LogLoss)
        # evals_result structure: {'validation': {'logloss': [0.1, 0.09, ...]}}
        if "validation" in evals_result and "logloss" in evals_result["validation"]:
            history = evals_result["validation"]["logloss"]
            best_idx = model.best_iteration
            # If best_iteration is beyond the length (can happen if early stopping didn't trigger), cap it
            if best_idx >= len(history):
                best_idx = len(history) - 1
            best_score = history[best_idx]
            logger.info(
                f"Stream {stream_type} Best Validation LogLoss: {best_score:.16f}"
            )
        else:
            logger.info(
                f"Stream {stream_type} Training completed. (No validation metrics captured)"
            )

        # Store model in instance
        if stream_type == "A":
            self.model_a = model
        else:
            self.model_b = model

        return model

    def predict_stream(self, model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
        """
        Generates raw probability predictions for a specific stream.

        Args:
            model (xgb.Booster): The trained model.
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Probability scores (0.0 to 1.0).
        """
        if X.empty:
            return np.array([])

        dtest = xgb.DMatrix(X)
        # iteration_range handles using the best iteration if early stopping was used
        preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        return preds

    def save_models(self, suffix: str = ""):
        """
        Saves the trained models to disk.

        Args:
            suffix (str): Optional suffix for filenames (e.g., hash or timestamp).
        """
        if self.model_a:
            path_a = os.path.join(self.models_dir, f"model_stream_a{suffix}.json")
            self.model_a.save_model(path_a)
            logger.info(f"Saved Stream A model to {path_a}")

        if self.model_b:
            path_b = os.path.join(self.models_dir, f"model_stream_b{suffix}.json")
            self.model_b.save_model(path_b)
            logger.info(f"Saved Stream B model to {path_b}")

    def load_models(self, suffix: str = ""):
        """
        Loads models from disk.

        Args:
            suffix (str): Optional suffix to match filenames.
        """
        path_a = os.path.join(self.models_dir, f"model_stream_a{suffix}.json")
        if os.path.exists(path_a):
            self.model_a = xgb.Booster()
            self.model_a.load_model(path_a)
            logger.info(f"Loaded Stream A model from {path_a}")
        else:
            logger.warning(f"Stream A model not found at {path_a}")

        path_b = os.path.join(self.models_dir, f"model_stream_b{suffix}.json")
        if os.path.exists(path_b):
            self.model_b = xgb.Booster()
            self.model_b.load_model(path_b)
            logger.info(f"Loaded Stream B model from {path_b}")
        else:
            logger.warning(f"Stream B model not found at {path_b}")
