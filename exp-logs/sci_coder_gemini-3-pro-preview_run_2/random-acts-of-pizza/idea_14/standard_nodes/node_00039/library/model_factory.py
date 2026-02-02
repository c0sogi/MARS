from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import (
    LR_PARAM_GRID,
    BAGGING_N_ESTIMATORS,
    BAGGING_MAX_SAMPLES,
    SEED,
)


def create_bagged_logistic_ensemble(
    n_estimators=BAGGING_N_ESTIMATORS,
    max_samples=BAGGING_MAX_SAMPLES,
    random_state=SEED,
    n_jobs=-1,
):
    """
    Creates a BaggingClassifier with a LogisticRegression base estimator.

    Args:
        n_estimators (int): Number of base estimators in the ensemble.
        max_samples (float): The number of samples to draw from X to train each base estimator.
        random_state (int): Seed used by the random number generator.
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        BaggingClassifier: The configured ensemble model.
    """
    # Initialize base estimator
    # We set defaults here, but they will be overridden by GridSearchCV if specified in the grid.
    # solver='liblinear' is chosen as per strategy for robust performance on high-dim/sparse data.
    base_estimator = LogisticRegression(
        random_state=random_state, solver="liblinear", max_iter=1000
    )

    # Create BaggingClassifier
    # Using 'estimator' parameter as 'base_estimator' is removed in recent sklearn versions
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model


def get_hyperparameter_grid():
    """
    Prepares the hyperparameter grid for GridSearchCV.
    Maps the LogisticRegression parameters to the BaggingClassifier's estimator namespace.

    Returns:
        dict: Parameter grid compatible with BaggingClassifier tuning.
    """
    grid = {}
    for param, values in LR_PARAM_GRID.items():
        # Map 'param' (e.g., 'C') to 'estimator__param' (e.g., 'estimator__C')
        # This tells GridSearchCV to update the parameter inside the base LogisticRegression
        grid[f"estimator__{param}"] = values

    return grid
