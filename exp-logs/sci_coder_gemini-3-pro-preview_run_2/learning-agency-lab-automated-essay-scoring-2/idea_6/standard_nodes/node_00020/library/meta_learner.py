import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.optimize as optimize
from library.config import Config
from library.utils import quadratic_weighted_kappa


class MetaLearner:
    """
    Implements the Non-Linear Stacking Meta-Learner using LightGBM.
    Combines predictions from Semantic and Lexical branches with Mechanics features.
    """

    def __init__(self):
        """
        Initializes the MetaLearner with parameters from Config.
        """
        self.params = Config.LGB_PARAMS.copy()
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with early stopping.

        Args:
            X_train (pd.DataFrame or np.array): Training features.
            y_train (np.array): Training targets.
            X_val (pd.DataFrame or np.array): Validation features.
            y_val (np.array): Validation targets.
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.params["early_stopping_rounds"]),
            lgb.log_evaluation(period=100),
        ]

        print(f"Training Meta-Learner on {len(X_train)} samples...")

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log final validation metric
        if self.model.best_score:
            val_rmse = self.model.best_score["valid"]["rmse"]
            print(f"Best Validation RMSE: {val_rmse}")

        return self

    def predict(self, X):
        """
        Generates continuous predictions.

        Args:
            X (pd.DataFrame or np.array): Features to predict on.

        Returns:
            np.array: Continuous score predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path):
        """
        Saves the trained model to disk.

        Args:
            path (str): Destination file path.
        """
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Meta-learner saved to {path}")

    def load(self, path):
        """
        Loads a trained model from disk.

        Args:
            path (str): Source file path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = joblib.load(path)
        print(f"Meta-learner loaded from {path}")
        return self


def apply_thresholds(y_pred, thresholds):
    """
    Maps continuous predictions to integers [1, 6] based on thresholds.

    Args:
        y_pred (np.array): Continuous predictions.
        thresholds (list or np.array): 5 threshold values separating the 6 classes.

    Returns:
        np.array: Integer predictions.
    """
    # Ensure thresholds are sorted
    thresholds = np.sort(thresholds)

    # Digitizing:
    # bins: [-inf, t0, t1, t2, t3, t4, inf]
    # indices: 0 (score 1), 1 (score 2), ..., 5 (score 6)
    # We add 1 to the result to get 1-based scoring.
    bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
    return np.digitize(y_pred, bins) - 1 + 1


def optimize_thresholds(y_true, y_pred):
    """
    Optimizes decision boundaries using Nelder-Mead to maximize QWK.

    Args:
        y_true (np.array): Ground truth scores.
        y_pred (np.array): Continuous predictions from the meta-learner.

    Returns:
        np.array: Optimized thresholds.
    """
    # Initial guess: standard rounding boundaries
    # 1.5 separates 1 and 2, 2.5 separates 2 and 3, etc.
    initial_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    # Objective function to minimize (Negative QWK)
    def negative_qwk(thresholds):
        # Constraint: Thresholds must be ordered.
        # While Nelder-Mead doesn't enforce constraints, QWK will naturally be poor
        # if thresholds are unordered. We can sort them inside apply_thresholds.
        preds_int = apply_thresholds(y_pred, thresholds)
        score = quadratic_weighted_kappa(y_true, preds_int)
        return -score

    print("Optimizing thresholds using Nelder-Mead...")
    result = optimize.minimize(
        negative_qwk,
        initial_thresholds,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-3},
    )

    best_thresholds = np.sort(result.x)
    best_score = -result.fun

    print(f"Optimization Complete. Best QWK: {best_score}")
    print(f"Optimized Thresholds: {best_thresholds}")

    return best_thresholds
