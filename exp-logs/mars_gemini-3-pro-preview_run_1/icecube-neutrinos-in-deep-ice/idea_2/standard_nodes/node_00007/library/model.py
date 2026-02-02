import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import joblib
from library.config import LGBM_PARAMS, WORKING_DIR


class DirectionalLGBM:
    """
    A wrapper class for three LightGBM regressors to predict the x, y, and z
    components of a neutrino's direction vector.
    """

    def __init__(self):
        # Initialize three independent regressors with the configuration parameters
        self.models = {
            "x": lgb.LGBMRegressor(**LGBM_PARAMS),
            "y": lgb.LGBMRegressor(**LGBM_PARAMS),
            "z": lgb.LGBMRegressor(**LGBM_PARAMS),
        }
        self.model_path = os.path.join(WORKING_DIR, "lgbm_models.pkl")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the three regressors (x, y, z) with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets with columns 'x', 'y', 'z'.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame, optional): Validation targets with columns 'x', 'y', 'z'.
        """
        for component in ["x", "y", "z"]:
            print(f"Training model for component: {component}")
            model = self.models[component]

            # Setup callbacks for early stopping and logging
            callbacks = []
            eval_set = None

            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[component])]
                # Stop if validation score doesn't improve for 50 rounds
                callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
                # Suppress iteration logging
                callbacks.append(lgb.log_evaluation(period=0))

            model.fit(
                X_train,
                y_train[component],
                eval_set=eval_set,
                eval_metric="mse",
                callbacks=callbacks,
            )

            # Print validation metric
            if X_val is not None and y_val is not None:
                # best_score_ structure: {'valid_0': {'l2': 0.12345}}
                # 'l2' corresponds to 'mse' in LightGBM
                valid_key = list(model.best_score_.keys())[0]
                metric_key = list(model.best_score_[valid_key].keys())[0]
                score = model.best_score_[valid_key][metric_key]
                print(f"Component {component} Validation MSE: {score}")

    def predict(self, X):
        """
        Predicts the unit direction vector for the given features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Array of shape (N, 3) containing normalized (x, y, z) vectors.
        """
        preds = {}
        for component in ["x", "y", "z"]:
            preds[component] = self.models[component].predict(X)

        # Stack predictions to form vectors: shape (N, 3)
        vectors = np.stack([preds["x"], preds["y"], preds["z"]], axis=1)

        # Normalize vectors to unit length
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-8, norms)
        normalized_vectors = vectors / norms

        return normalized_vectors

    def save(self, path=None):
        """
        Saves the trained models to disk.
        """
        if path is None:
            path = self.model_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models, path)
        print(f"Models saved to {path}")

    def load(self, path=None):
        """
        Loads trained models from disk.
        """
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        self.models = joblib.load(path)
        print(f"Models loaded from {path}")
