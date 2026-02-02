import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger


class ModelFactory:
    """
    Factory class for creating the Hook-Augmented Multi-Field Asymmetric
    Dual-Backbone Ensemble (HAMF-ADBE) classifier.
    """

    def __init__(self):
        self.logger = setup_logger("model_factory")

    def get_hyperparameter_grid(self):
        """
        Returns the hyperparameter grid for GridSearchCV, formatted for the
        BaggingClassifier wrapping a LogisticRegression estimator.

        Returns:
            dict: A dictionary with keys prefixed by 'estimator__' mapping to lists of values.
        """
        raw_grid = Config.PARAM_GRID
        # Prefix parameters with 'estimator__' to target the inner LogisticRegression
        # within the BaggingClassifier during GridSearchCV
        formatted_grid = {f"estimator__{key}": value for key, value in raw_grid.items()}
        return formatted_grid

    def create_classifier(self, params=None):
        """
        Creates and returns the BaggingClassifier ensemble.

        Args:
            params (dict, optional): Hyperparameters for the model.
                                     Can be keyed with 'estimator__C' (GridSearch style)
                                     or just 'C' (direct style).

        Returns:
            BaggingClassifier: The configured ensemble model ready for training.
        """
        # Default parameters for the base estimator (Ridge Logistic Regression)
        base_params = {
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": Config.SEED,
        }

        # Parse provided params to update defaults
        if params:
            for key, value in params.items():
                # Handle keys from GridSearchCV (e.g., 'estimator__C')
                if key.startswith("estimator__"):
                    clean_key = key.replace("estimator__", "")
                    base_params[clean_key] = value
                # Handle direct keys (e.g., 'C')
                elif key in ["C", "class_weight"]:
                    base_params[key] = value

        # Set default C if not provided (standard L2 regularization)
        if "C" not in base_params:
            base_params["C"] = 1.0

        self.logger.info(
            f"Initializing Base Estimator (LogisticRegression) with params: {base_params}"
        )

        # Instantiate Base Estimator
        base_clf = LogisticRegression(**base_params)

        # Instantiate Bagging Ensemble
        # We use the base estimator defined above
        # n_jobs=-1 allows using all CPU cores for parallel bagging
        bagging_clf = BaggingClassifier(
            estimator=base_clf,
            n_estimators=Config.N_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=-1,
        )

        return bagging_clf
