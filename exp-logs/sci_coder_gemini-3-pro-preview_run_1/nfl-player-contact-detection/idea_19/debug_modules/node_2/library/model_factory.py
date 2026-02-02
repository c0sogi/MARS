import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import Config
from library.utils import setup_logging

# Setup logging
setup_logging()


def get_estimator(model_type, params=None):
    """
    Factory function to create model instances based on type.
    Encapsulates the creation of the heterogeneous model ensemble components.

    Args:
        model_type (str): 'lgbm', 'xgb'.
        params (dict, optional): Hyperparameters. If None, defaults from Config are used.

    Returns:
        model: An sklearn-compatible classifier instance.
    """
    # Load defaults from Config if params not provided
    if params is None:
        if model_type == "lgbm":
            params = Config.LGBM_PARAMS.copy()
        elif model_type == "xgb":
            params = Config.XGB_PARAMS.copy()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Instantiate the requested model
    if model_type == "lgbm":
        # LightGBM: Leaf-wise growth, handles imbalance via is_unbalance=True
        model = lgb.LGBMClassifier(**params)
    elif model_type == "xgb":
        # XGBoost: Level-wise growth, scale_pos_weight managed by caller/params
        model = xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


class EnsemblePredictor:
    """
    Aggregates predictions from multiple expert models (Tri-Model Expert Ensemble).
    Implements unweighted averaging of probabilities.
    """

    def __init__(self, models):
        """
        Args:
            models (list): List of trained classifier instances (sklearn-compatible).
        """
        self.models = models

    def predict_proba(self, X):
        """
        Returns the averaged probability of the positive class across all experts.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.

        Returns:
            np.array: Probabilities of class 1.
        """
        preds = []
        for model in self.models:
            # predict_proba returns [prob_0, prob_1]
            # We take the second column (index 1) for the positive class
            try:
                p = model.predict_proba(X)[:, 1]
            except Exception as e:
                # Fallback for models that might behave differently or if X format issues arise
                print(f"Error predicting with model {type(model).__name__}: {e}")
                raise e
            preds.append(p)

        if not preds:
            return np.zeros(X.shape[0])

        # Stack and average
        preds_stack = np.column_stack(preds)
        avg_preds = np.mean(preds_stack, axis=1)
        return avg_preds

    def predict(self, X, threshold=0.5):
        """
        Returns binary class predictions based on a specific threshold.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.
            threshold (float): Decision threshold (optimized on validation set).

        Returns:
            np.array: Binary predictions (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
