import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger("model_builder")


def create_classifier(
    C: float = 1.0,
    class_weight=None,
    n_estimators: int = Config.BAGGING_N_ESTIMATORS,
    random_state: int = Config.RANDOM_SEED,
) -> BaggingClassifier:
    """
    Creates a Bagged Ensemble of Logistic Regression classifiers.

    This architecture combines the high-bias linear core of Logistic Regression
    (using L2 regularization) with the variance reduction properties of Bagging.
    This is specifically designed to handle the fused feature space (Text Embeddings
    + Topic Vectors + Metadata) robustly.

    Args:
        C (float): Inverse of regularization strength for the base LogisticRegression.
                   Smaller values specify stronger regularization.
        class_weight (dict or 'balanced' or None): Weights associated with classes
                                                   to handle imbalance.
        n_estimators (int): The number of base estimators in the ensemble.
        random_state (int): Controls the random resampling and base estimator seeding
                            for reproducibility.

    Returns:
        BaggingClassifier: The configured ensemble model ready for training or tuning.
    """
    # Initialize the base estimator: Logistic Regression with L2 regularization
    # We use 'lbfgs' solver which is robust and supports L2.
    # max_iter is increased to 1000 to ensure convergence on the fused feature set.
    base_estimator = LogisticRegression(
        C=C,
        class_weight=class_weight,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    # Initialize the Bagging Classifier
    # We wrap the base linear model to reduce variance and improve stability.
    # n_jobs=-1 allows parallel training of the ensemble members.
    clf = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    return clf
