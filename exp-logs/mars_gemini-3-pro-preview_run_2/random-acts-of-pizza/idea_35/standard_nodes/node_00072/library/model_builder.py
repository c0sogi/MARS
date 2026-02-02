import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.oasf_transformers import OASFPreprocessor


def get_model_pipeline(
    n_estimators=Config.N_ESTIMATORS, random_state=Config.RANDOM_SEED
):
    """
    Constructs the OASF Pipeline and the corresponding hyperparameter grid.

    The pipeline consists of:
    1. OASFPreprocessor: Handles feature splitting, orthogonalization, PCA, and fusion.
    2. Classifier: BaggingClassifier wrapping a LogisticRegression(Ridge) estimator.

    Args:
        n_estimators (int): Number of base estimators for Bagging.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (pipeline, param_grid)
            - pipeline: The scikit-learn Pipeline object.
            - param_grid: Dictionary defining the hyperparameter search space.
    """

    # 1. Preprocessor
    # Initializes the OASF strategy transformer using configuration constants
    preprocessor = OASFPreprocessor(
        anchor_dim=Config.ANCHOR_DIM,
        aux_dim=Config.AUX_DIM,
        pca_components=Config.PCA_COMPONENTS,
        random_state=random_state,
    )

    # 2. Base Estimator
    # Logistic Regression with L2 penalty (Ridge).
    # We use 'lbfgs' which is standard for L2 and increase max_iter to ensure
    # convergence on the fused embedding space.
    base_estimator = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000, random_state=random_state
    )

    # 3. Ensemble Classifier
    # Bagging reduces variance of the linear estimators.
    # We set n_jobs=1 here to avoid nested parallelism issues when the outer
    # Cross-Validation/GridSearch loop is parallelized.
    classifier = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )

    # 4. Pipeline Construction
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    # 5. Parameter Grid Definition
    # We must map the generic parameter names in Config.PARAM_GRID to the
    # specific nested structure of the pipeline: classifier -> estimator -> parameter
    raw_grid = Config.PARAM_GRID
    param_grid = {}

    for key, values in raw_grid.items():
        # Construct the double-underscore path required by scikit-learn
        grid_key = f"classifier__estimator__{key}"
        param_grid[grid_key] = values

    return pipeline, param_grid
