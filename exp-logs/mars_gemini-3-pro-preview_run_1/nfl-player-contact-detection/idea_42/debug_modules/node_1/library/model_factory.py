import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import LGBM_PARAMS, XGB_PARAMS, SEED


def get_model(model_type, random_state=SEED):
    """
    Factory function to instantiate a model based on the provided type and configuration.

    Args:
        model_type (str): Type of model to create. Options: 'lgbm', 'xgb'.
        random_state (int): Seed for reproducibility.

    Returns:
        model: An instance of LGBMClassifier or XGBClassifier.
    """
    if model_type == "lgbm":
        params = LGBM_PARAMS.copy()
        params["random_state"] = random_state
        # Instantiate LightGBM Classifier
        model = lgb.LGBMClassifier(**params)
        return model

    elif model_type == "xgb":
        params = XGB_PARAMS.copy()
        params["random_state"] = random_state
        # Instantiate XGBoost Classifier
        model = xgb.XGBClassifier(**params)
        return model

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported types: 'lgbm', 'xgb'"
        )


def save_model(model, filepath):
    """
    Saves a trained model to disk using joblib.

    Args:
        model: The trained model object.
        filepath (str): Path where the model should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Loads a model from disk using joblib.

    Args:
        filepath (str): Path to the saved model file.

    Returns:
        model: The loaded model object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")

    model = joblib.load(filepath)
    return model


class EnsemblePredictor:
    """
    Handles inference for the Unified Heterogeneous Dual-Ensemble.
    Loads multiple trained models and computes the unweighted average of their predictions.
    """

    def __init__(self, model_paths):
        """
        Initialize the predictor by loading models from the specified paths.

        Args:
            model_paths (list of str): List of file paths to the saved models (e.g., .joblib files).
        """
        self.models = []
        print(
            f"[{self.__class__.__name__}] Loading {len(model_paths)} models for ensemble..."
        )
        for path in model_paths:
            try:
                model = load_model(path)
                self.models.append(model)
                print(f" - Loaded model from {path}")
            except Exception as e:
                print(f" - Error loading model from {path}: {e}")
                raise e

    def predict_proba(self, X):
        """
        Generate ensemble probabilities.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.

        Returns:
            np.array: Array of probabilities for the positive class (contact=1).
        """
        if not self.models:
            raise RuntimeError("No models loaded in EnsemblePredictor.")

        # Collect predictions from all models
        # We assume binary classification and take the probability of class 1
        preds = []
        for i, model in enumerate(self.models):
            # predict_proba returns [prob_0, prob_1]
            p = model.predict_proba(X)[:, 1]
            preds.append(p)

        # Stack and average
        preds_stack = np.vstack(preds)
        avg_preds = np.mean(preds_stack, axis=0)

        return avg_preds

    def predict(self, X, threshold=0.5):
        """
        Generate binary class predictions based on a threshold.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.
            threshold (float): Decision threshold.

        Returns:
            np.array: Binary predictions (0 or 1).
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
