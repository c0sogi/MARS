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

    def get_bagged_ensemble(
        self, C: float = 1.0, class_weight=None
    ) -> BaggingClassifier:
        """
        Constructs a BaggingClassifier with a LogisticRegression base estimator.

        Args:
            C (float): Inverse of regularization strength for Logistic Regression.
                       Smaller values specify stronger regularization.
            class_weight (str or None): Weights associated with classes in the form {class_label: weight}.
                                        If 'balanced', uses the values of y to automatically adjust weights.

        Returns:
            BaggingClassifier: The configured ensemble model ready for training.
        """
        # Base Estimator: Logistic Regression
        # We use 'lbfgs' as it is a robust solver for multiclass and binary problems.
        # max_iter is set to 1000 to ensure convergence on the fused feature space.
        # n_jobs is set to 1 because parallelism is handled at the Bagging level.
        base_estimator = LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=1000,
            random_state=Config.SEED,
            n_jobs=1,
        )

        # Bagging Ensemble
        # Wraps the high-bias linear core to reduce variance via bootstrapping.
        # n_estimators is retrieved from Config (default 20).
        # n_jobs utilizes the available vCPUs defined in Config.
        ensemble = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=Config.N_JOBS,
            bootstrap=True,
            bootstrap_features=False,
        )

        return ensemble
