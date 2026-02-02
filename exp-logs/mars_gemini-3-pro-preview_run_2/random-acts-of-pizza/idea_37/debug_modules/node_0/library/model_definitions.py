import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config


def get_hyperparameter_grid():
    """
    Retrieves the hyperparameter search space for the Logistic Regression model
    as defined in the configuration.

    Returns:
        dict: A dictionary where keys are parameter names and values are lists
              of values to explore during grid search.
    """
    return Config.LR_PARAM_GRID


def get_bagged_lr_pipeline(
    C=1.0,
    class_weight=None,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    n_estimators=Config.N_ESTIMATORS_BAGGING,
    random_state=Config.RANDOM_SEED,
    n_jobs=-1,
):
    """
    Constructs a BaggingClassifier wrapping a LogisticRegression estimator.
    This serves as the base architecture for both the Parsimonious (Pipeline A)
    and Augmented (Pipeline B) views in the TMVCE strategy.

    Args:
        C (float): Inverse of regularization strength; smaller values specify
                   stronger regularization. Defaults to 1.0.
        class_weight (str, dict, or None): Weights associated with classes.
                                           'balanced' or None. Defaults to None.
        penalty (str): The norm of the penalty ('l2'). Defaults to 'l2'.
        solver (str): Algorithm to use in the optimization problem. Defaults to 'lbfgs'.
        max_iter (int): Maximum number of iterations for the solver to converge.
                        Controls training duration. Defaults to 1000.
        n_estimators (int): The number of base estimators in the ensemble.
                            Defaults to Config.N_ESTIMATORS_BAGGING.
        random_state (int): Seed used by the random number generator.
                            Defaults to Config.RANDOM_SEED.
        n_jobs (int): The number of jobs to run in parallel. Defaults to -1 (all CPUs).

    Returns:
        BaggingClassifier: An instantiated ensemble model ready for training.
    """
    # Initialize the base estimator (Logistic Regression)
    # We strictly enforce the random_state here for reproducibility of the base learner
    base_estimator = LogisticRegression(
        C=C,
        class_weight=class_weight,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Initialize the Bagging Classifier
    # Wraps the base estimator to create a diverse ensemble via bootstrapping
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model
