import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("Model", os.path.join(Config.WORKING_DIR, "model.log"))


def get_bagged_ensemble(
    C: float = 1.0,
    class_weight=None,
    n_estimators: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BaggingClassifier:
    """
    Constructs a BaggingClassifier with a LogisticRegression base estimator.

    Args:
        C (float): Inverse of regularization strength for the base LogisticRegression.
        class_weight (dict or 'balanced' or None): Weights associated with classes.
        n_estimators (int): The number of base estimators in the ensemble.
        random_state (int): Controls the random resampling and base estimator.
        n_jobs (int): The number of jobs to run in parallel for both fit and predict.

    Returns:
        BaggingClassifier: The configured ensemble model ready for training.
    """
    # Base Learner: Logistic Regression
    # We use 'liblinear' as it is well-suited for high-dimensional sparse/dense data
    # and supports both L1 and L2 regularization (though default is L2).
    base_lr = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        random_state=random_state,
    )

    # Bagging Wrapper
    # This reduces variance by training multiple base learners on bootstrap samples.
    ensemble = BaggingClassifier(
        estimator=base_lr,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return ensemble


def tune_ensemble_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict,
    n_estimators: int = 20,
    random_state: int = 42,
):
    """
    Performs a grid search to optimize the hyperparameters of the Bagged Ensemble.
    Crucially, this tunes the parameters of the base learner (C, class_weight)
    by evaluating the performance of the *full ensemble*.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation labels.
        param_grid (dict): Dictionary with parameters names (`C`, `class_weight`) as keys
                           and lists of parameter settings to try as values.
        n_estimators (int): Number of estimators for the bagging ensemble.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (best_model, best_params, best_score)
            - best_model (BaggingClassifier): The fitted model with the best performance.
            - best_params (dict): The hyperparameters corresponding to the best model.
            - best_score (float): The ROC AUC score of the best model on the validation set.
    """
    logger.info("Starting hyperparameter tuning for Bagged Ensemble...")

    grid = ParameterGrid(param_grid)
    best_score = -float("inf")
    best_model = None
    best_params = {}

    for params in grid:
        C = params.get("C", 1.0)
        class_weight = params.get("class_weight", None)

        # Construct the ensemble with current hyperparameters
        model = get_bagged_ensemble(
            C=C,
            class_weight=class_weight,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        # Fit the full ensemble
        model.fit(X_train, y_train)

        # Evaluate on validation set
        # predict_proba returns [n_samples, n_classes], we take the probability of class 1
        val_probs = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_probs)

        # Log full precision score
        logger.info(f"Params: {params} | Val AUC: {score}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    logger.info(f"Tuning complete. Best AUC: {best_score} | Best Params: {best_params}")

    return best_model, best_params, best_score
