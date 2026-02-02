import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config
from library.utils import setup_logger


class ModelBuilder:
    """
    Constructs the Asymmetric Multi-View Bagged Linear Ensemble (AMBLE).
    Manages model architecture definition and hyperparameter grid retrieval.
    """

    def __init__(self):
        """Initialize the ModelBuilder with a logger."""
        self.logger = setup_logger("ModelBuilder")

    def get_hyperparameter_grid(self) -> dict:
        """
        Returns the hyperparameter grid defined in the configuration.

        Returns:
            dict: A dictionary where keys are parameter names and values are lists of values to search.
        """
        return Config.GRID_SEARCH_PARAMS

    from sklearn.pipeline import Pipeline
    from library.feature_engineering import ViewTransformer

    def get_pipeline(self) -> Pipeline:
        """
        Constructs a Pipeline containing the ViewTransformer and the BaggingClassifier.
        This ensures preprocessing is part of the CV loop (Lesson 47).
        """
        # Base Estimator: Logistic Regression
        # Hyperparameters (C, class_weight) will be set via GridSearchCV using the prefix
        base_estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=Config.SEED,
            n_jobs=1,
        )

        # Bagging Ensemble
        ensemble = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=Config.N_JOBS,
            bootstrap=True,
            bootstrap_features=False,
        )

        # Pipeline
        pipeline = Pipeline(
            [("preprocessor", ViewTransformer()), ("ensemble", ensemble)]
        )

        return pipeline
