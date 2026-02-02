import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config
from library.utils import set_seed


def create_bagged_ensemble(
    c_param=Config.LR_C,
    max_iter=Config.LR_MAX_ITER,
    n_estimators=Config.BAGGING_N_ESTIMATORS,
    max_samples=Config.BAGGING_MAX_SAMPLES,
    max_features=Config.BAGGING_MAX_FEATURES,
    random_seed=Config.RANDOM_SEED,
):
    """
    Constructs a Bagged Ensemble of Logistic Regression classifiers.

    This architecture combines the high-bias, low-variance properties of linear models
    with the variance-reduction properties of bagging. It is designed to operate on
    the fused feature space (Projected Text + Interaction Metadata).

    Args:
        c_param (float): Inverse of regularization strength for the LogisticRegression base estimator.
                         Smaller values specify stronger regularization.
        max_iter (int): Maximum number of iterations for the solver to converge.
        n_estimators (int): The number of base estimators in the ensemble.
        max_samples (float): The fraction of samples to draw from X to train each base estimator.
        max_features (float): The fraction of features to draw from X to train each base estimator.
        random_seed (int): Seed used by the random number generator for reproducibility.

    Returns:
        BaggingClassifier: The configured ensemble model ready for training.
    """
    # Ensure global reproducibility for this model construction
    set_seed(random_seed)

    # 1. Base Estimator: Logistic Regression
    # We use class_weight='balanced' to automatically adjust weights inversely proportional
    # to class frequencies in the input data, addressing the ~3:1 imbalance.
    base_lr = LogisticRegression(
        C=c_param,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_seed,
        n_jobs=1,  # Parallelism is handled at the Bagging level
    )

    # 2. Ensemble: BaggingClassifier
    # Bagging allows us to train multiple linear models on different subsets of data/features.
    # This helps stabilize predictions, especially given the potential noise in the
    # projected text embeddings and polynomial interaction features.
    ensemble = BaggingClassifier(
        estimator=base_lr,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=-1,  # Utilize all available cores for parallel training
        verbose=0,
    )

    return ensemble
