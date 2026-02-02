from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config


class ModelFactory:
    """
    Factory class for creating the classifier and retrieving its hyperparameter grid.
    Implements the classification head for the Affect-Augmented Asymmetric Early Fusion (AAAEF) strategy.
    """

    @staticmethod
    def get_classifier():
        """
        Constructs the BaggingClassifier wrapping a LogisticRegression estimator.

        Strategy:
        - Base: Logistic Regression (Linear model suitable for high-dimensional fused features).
        - Ensemble: Bagging (Reduces variance and improves stability of probability estimates).

        Returns:
            sklearn.ensemble.BaggingClassifier: The configured ensemble model.
        """
        # Base estimator: Logistic Regression
        # We use 'lbfgs' as it is robust and standard for this version of scikit-learn.
        # max_iter is increased to ensure convergence given the dimensionality of fused features.
        base_estimator = LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=Config.SEED
        )

        # Bagging Ensemble
        # Wraps the logistic regression.
        # 'estimator' parameter is used for scikit-learn >= 1.2
        # n_jobs=-1 allows parallel training of the ensemble members.
        clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.N_BAGGING_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=-1,
        )

        return clf

    @staticmethod
    def get_hyperparameter_grid():
        """
        Returns the hyperparameter search space defined in the configuration.

        The grid keys (e.g., 'bagging__estimator__C') correspond to the structure
        expected when this classifier is used as a step named 'bagging' in a Pipeline.

        Returns:
            dict: The parameter grid.
        """
        return Config.LR_PARAM_GRID
