from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config


def get_pls_transformer(n_components: int):
    """
    Returns a PLSRegression model configured to act as a supervised dimensionality reduction transformer.
    It projects X into a latent space that maximizes covariance with Y.

    Args:
        n_components (int): Number of latent components to keep.

    Returns:
        PLSRegression: The configured PLS model.
    """
    # scale=True (default) ensures X and Y are centered and scaled before projection.
    return PLSRegression(n_components=n_components, scale=True, copy=True)


def get_scaler(output_distribution: str = "normal"):
    """
    Returns a QuantileTransformer for RankGauss scaling of numerical metadata.

    Args:
        output_distribution (str): 'normal' for RankGauss, 'uniform' otherwise.

    Returns:
        QuantileTransformer: The configured scaler.
    """
    return QuantileTransformer(
        output_distribution=output_distribution, random_state=Config.SEED
    )


def get_standard_scaler():
    """
    Returns a StandardScaler. Useful for scaling PLS components to unit variance
    before feature fusion.

    Returns:
        StandardScaler: The configured standard scaler.
    """
    return StandardScaler()


def get_classifier(
    C: float = 1.0,
    class_weight=None,
    n_estimators: int = 10,
    n_jobs: int = 1,
    random_state: int = 42,
):
    """
    Returns a BaggingClassifier wrapping a LogisticRegression model.

    Args:
        C (float): Inverse of regularization strength for Logistic Regression.
        class_weight (str or dict or None): Class weights ('balanced' or None).
        n_estimators (int): Number of base estimators in the ensemble.
        n_jobs (int): Number of jobs to run in parallel.
        random_state (int): Seed for reproducibility.

    Returns:
        BaggingClassifier: The ensemble classifier.
    """
    # Inner estimator: Logistic Regression
    # We use 'lbfgs' as it is robust and standard for this dimensionality.
    # max_iter is increased to ensure convergence.
    base_clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )

    # Outer estimator: Bagging
    # Reduces variance of the linear model.
    bagging_clf = BaggingClassifier(
        estimator=base_clf,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return bagging_clf
