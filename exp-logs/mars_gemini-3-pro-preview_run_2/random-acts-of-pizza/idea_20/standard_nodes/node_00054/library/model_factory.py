from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config


def create_classifier(
    n_estimators=Config.BAGGING_N_ESTIMATORS,
    C=1.0,
    class_weight=None,
    random_state=Config.SEED,
    n_jobs=Config.N_JOBS,
    max_iter=1000,
):
    """
    Constructs a Bagged Ensemble of Logistic Regression Classifiers.

    This architecture combines the high-bias, stable nature of Logistic Regression (L2 Ridge)
    with the variance-reduction properties of Bagging. It is designed to operate on the
    fused high-dimensional feature space (SBERT + PLS + Metadata).

    Args:
        n_estimators (int): Number of base estimators in the ensemble.
        C (float): Inverse of regularization strength for the Logistic Regression base estimator.
                   Smaller values specify stronger regularization.
        class_weight (dict or 'balanced', optional): Weights associated with classes.
        random_state (int): Seed used by the random number generator for reproducibility.
        n_jobs (int): The number of jobs to run in parallel for both fit and predict.
        max_iter (int): Maximum number of iterations for the solver to converge.

    Returns:
        sklearn.ensemble.BaggingClassifier: The configured ensemble model.
    """
    # Base Estimator: Logistic Regression with L2 Regularization
    # We use 'lbfgs' solver which supports L2 and is robust for dense features.
    # max_iter is increased to ensure convergence on the fused feature set.
    base_estimator = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        penalty="l2",
        max_iter=max_iter,
        random_state=random_state,
    )

    # Ensemble: Bagging Classifier
    # Wraps the linear model to reduce variance and improve stability.
    clf = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=1.0,  # Train on size of input (with replacement)
        max_features=1.0,  # Use all features
        bootstrap=True,  # Standard bagging
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return clf
