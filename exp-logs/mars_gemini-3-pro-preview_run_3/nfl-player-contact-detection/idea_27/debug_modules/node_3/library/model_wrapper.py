import numpy as np
import pandas as pd
import xgboost as xgb
import os
from library.config import Config


class DualStreamXGBoost:
    """
    Wrapper for the Invariant-Physics Temporal Pyramid Dual-Stream GBDT.
    Manages two independent XGBoost models:
    - Stream A: Player-Player Interactions (Relational Dynamics)
    - Stream B: Player-Ground Impacts (Invariant Ego-Centric Kinematics)
    """

    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.params_a = Config.XGB_PARAMS_A
        self.params_b = Config.XGB_PARAMS_B
        self.undersample_ratio = Config.UNDERSAMPLE_RATIO

    def _undersample(self, X, y):
        """
        Performs Targeted Majority Undersampling.
        Retains 100% of positive class and subsamples negative class to the configured ratio.
        """
        # Ensure y is numpy array for indexing logic
        y_np = np.array(y) if not isinstance(y, np.ndarray) else y

        pos_indices = np.where(y_np == 1)[0]
        neg_indices = np.where(y_np == 0)[0]

        n_pos = len(pos_indices)

        if n_pos == 0:
            n_neg_keep = len(neg_indices)
        else:
            n_neg_keep = int(n_pos * self.undersample_ratio)

        # If we have fewer negatives than the target ratio, keep all negatives
        if n_neg_keep > len(neg_indices):
            n_neg_keep = len(neg_indices)

        # Randomly sample negatives
        np.random.seed(Config.SEED)
        neg_indices_sampled = np.random.choice(neg_indices, n_neg_keep, replace=False)

        # Combine and shuffle
        indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(indices)

        # Return subset, preserving DataFrame structure if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            return X.iloc[indices], y_np[indices]

        # Fallback for numpy arrays
        X_np = np.array(X) if not isinstance(X, np.ndarray) else X
        return X_np[indices], y_np[indices]

    def fit_stream(self, X_train, y_train, stream, X_val=None, y_val=None):
        """
        Trains the XGBoost model for a specific stream.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            stream (str): 'A' or 'B'.
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
        """
        if stream not in ["A", "B"]:
            raise ValueError("Stream must be 'A' or 'B'.")

        print(f"Training Stream {stream}...")

        # Apply Undersampling to Training Data
        print(f"Original Train Shape: {X_train.shape}, Positives: {np.sum(y_train)}")
        X_train_res, y_train_res = self._undersample(X_train, y_train)
        print(
            f"Resampled Train Shape: {X_train_res.shape}, Positives: {np.sum(y_train_res)}"
        )

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))

        # Select Parameters
        params = self.params_a if stream == "A" else self.params_b
        params = params.copy()
        params["base_score"] = 0.5

        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        if stream == "A":
            self.model_a = model
        else:
            self.model_b = model

        print(f"Stream {stream} training complete. Best Score: {model.best_score}")

    def predict_stream(self, X, stream):
        """
        Generates probabilities for a specific stream.

        Args:
            X (np.ndarray): Features.
            stream (str): 'A' or 'B'.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if stream == "A":
            model = self.model_a
        elif stream == "B":
            model = self.model_b
        else:
            raise ValueError("Stream must be 'A' or 'B'.")

        if model is None:
            raise RuntimeError(
                f"Model for Stream {stream} has not been trained or loaded."
            )

        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

    def save_models(self, base_path=Config.WORKING_DIR):
        """
        Saves both models to JSON files.
        """
        os.makedirs(base_path, exist_ok=True)

        if self.model_a:
            path_a = os.path.join(base_path, "model_stream_a.json")
            self.model_a.save_model(path_a)
            print(f"Stream A model saved to {path_a}")

        if self.model_b:
            path_b = os.path.join(base_path, "model_stream_b.json")
            self.model_b.save_model(path_b)
            print(f"Stream B model saved to {path_b}")

    def load_models(self, base_path=Config.WORKING_DIR):
        """
        Loads models from JSON files if they exist.
        """
        path_a = os.path.join(base_path, "model_stream_a.json")
        path_b = os.path.join(base_path, "model_stream_b.json")

        if os.path.exists(path_a):
            self.model_a = xgb.Booster()
            self.model_a.load_model(path_a)
            print(f"Stream A model loaded from {path_a}")

        if os.path.exists(path_b):
            self.model_b = xgb.Booster()
            self.model_b.load_model(path_b)
            print(f"Stream B model loaded from {path_b}")
