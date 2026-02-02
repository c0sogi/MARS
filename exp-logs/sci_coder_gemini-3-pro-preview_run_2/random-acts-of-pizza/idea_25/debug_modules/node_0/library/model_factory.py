import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config
from library.utils import setup_logger

# Setup logger
logger = setup_logger("model_factory")


class ModelFactory:
    """
    Factory class to create and configure the machine learning model architecture.
    Implements a Bagging Ensemble of Logistic Regression classifiers.
    """

    @staticmethod
    def get_classifier(
        c: float = Config.LR_C,
        class_weight=Config.LR_CLASS_WEIGHT,
        max_iter: int = Config.LR_MAX_ITER,
        solver: str = Config.LR_SOLVER,
        n_estimators: int = Config.BAGGING_N_ESTIMATORS,
        max_samples: float = Config.BAGGING_MAX_SAMPLES,
        max_features: float = Config.BAGGING_MAX_FEATURES,
        random_seed: int = Config.RANDOM_SEED,
    ) -> BaggingClassifier:
        """
        Constructs the BaggingClassifier wrapping a LogisticRegression estimator.

        Args:
            c (float): Inverse of regularization strength for Logistic Regression.
                       Smaller values specify stronger regularization.
            class_weight (str or dict): Weights associated with classes in the form {class_label: weight}.
                                        If not given, all classes are supposed to have weight one.
                                        The "balanced" mode uses the values of y to automatically adjust weights.
            max_iter (int): Maximum number of iterations taken for the solvers to converge.
            solver (str): Algorithm to use in the optimization problem.
            n_estimators (int): The number of base estimators in the ensemble.
            max_samples (float): The number (or fraction) of samples to draw from X to train each base estimator.
            max_features (float): The number (or fraction) of features to draw from X to train each base estimator.
            random_seed (int): Seed used by the random number generator.

        Returns:
            BaggingClassifier: The configured ensemble model ready for training.
        """

        # 1. Define the Base Estimator
        # Logistic Regression with L2 regularization (Ridge)
        # We set n_jobs=1 here because parallelization is handled by the Bagging wrapper
        base_estimator = LogisticRegression(
            C=c,
            class_weight=class_weight,
            max_iter=max_iter,
            solver=solver,
            random_state=random_seed,
            n_jobs=1,
        )

        # 2. Define the Bagging Ensemble
        # Wraps the linear model to reduce variance and improve generalization
        model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_seed,
            n_jobs=-1,  # Use all available cores for training the ensemble
            verbose=0,
        )

        logger.info(
            f"Model created: BaggingClassifier(n_estimators={n_estimators}) "
            f"wrapping LogisticRegression(C={c}, class_weight={class_weight}, solver={solver})"
        )

        return model
