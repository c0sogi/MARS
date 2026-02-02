import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from library import config, utils


class MetaLearner:
    """
    A wrapper class for the Ridge Regression meta-learner used in the stacking ensemble.
    Manages feature ordering and model persistence to ensure robust inference.
    """

    def __init__(self):
        self.model = Ridge(**config.META_PARAMS)
        self.feature_names = None

    def fit(self, base_preds: dict, y_true: np.ndarray):
        """
        Fits the Ridge regression model on the base model predictions.

        Args:
            base_preds (dict): Dictionary where keys are model names and values are
                               numpy arrays of predictions (e.g., {'xgb': ..., 'lgbm': ...}).
            y_true (np.ndarray): Array of ground truth target values.
        """
        # Convert dictionary to DataFrame to ensure consistent column ordering
        # This is critical for stacking to ensure the meta-learner maps weights to correct models
        X = pd.DataFrame(base_preds)
        self.feature_names = X.columns.tolist()

        print(f"Meta-Learner fitting on data shape: {X.shape}")
        self.model.fit(X, y_true)

        print("Meta-Learner Coefficients:")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"  {name}: {coef}")
        print(f"  Intercept: {self.model.intercept_}")

    def predict(self, base_preds: dict) -> np.ndarray:
        """
        Predicts using the fitted Ridge model.

        Args:
            base_preds (dict): Dictionary of base model predictions.

        Returns:
            np.ndarray: The ensemble predictions.
        """
        if self.feature_names is None:
            raise RuntimeError("MetaLearner must be fitted before prediction.")

        X = pd.DataFrame(base_preds)

        # Validate existence of all required columns
        missing_cols = [col for col in self.feature_names if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing base model predictions: {missing_cols}")

        # Reorder columns to match fit order
        X = X[self.feature_names]

        return self.model.predict(X)

    def save(self, path: str):
        """Saves the MetaLearner object to disk using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"MetaLearner saved to {path}")

    @staticmethod
    def load(path: str):
        """Loads a MetaLearner object from disk."""
        print(f"Loading MetaLearner from {path}")
        return joblib.load(path)


def train_meta_learner(base_preds: dict, y_true: np.ndarray):
    """
    Trains the meta-learner on the provided base model predictions and targets.

    Args:
        base_preds (dict): Dictionary of base model predictions (typically from the validation set).
        y_true (np.ndarray): True target values corresponding to the predictions.

    Returns:
        MetaLearner: The trained meta-learner instance.
    """
    utils.seed_everything(config.SEED)

    print("\n=== Training Meta-Learner ===")

    # Instantiate and fit
    meta_model = MetaLearner()
    meta_model.fit(base_preds, y_true)

    # Evaluate on the training set (which represents the hold-out validation set)
    # This provides the ensemble's performance metric on unseen data relative to base models
    ensemble_preds = meta_model.predict(base_preds)
    rmse = utils.compute_rmse(y_true, ensemble_preds)

    # Print full precision metric
    print(f"Meta-Learner Ensemble RMSE: {rmse}")

    # Save the model
    meta_model.save(config.MODEL_META_PATH)

    return meta_model


def predict_meta(meta_model: MetaLearner, base_preds: dict):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        meta_model (MetaLearner): The trained meta-learner object.
        base_preds (dict): Dictionary of base model predictions for the test set.

    Returns:
        np.ndarray: The final ensemble predictions.
    """
    print("\n=== Generating Ensemble Predictions ===")
    preds = meta_model.predict(base_preds)
    return preds
