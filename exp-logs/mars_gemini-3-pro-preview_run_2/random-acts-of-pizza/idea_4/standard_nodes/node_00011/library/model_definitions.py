import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from library.config import Config


def build_linear_branch(
    C=1.0,
    class_weight=None,
    penalty="l2",
    solver="liblinear",
    max_iter=2000,
    random_state=Config.SEED,
):
    """
    Constructs the Linear classification pipeline (Logistic Regression).

    Args:
        C (float): Inverse of regularization strength.
        class_weight (dict or 'balanced'): Weights associated with classes.
        penalty (str): Regularization norm.
        solver (str): Optimization algorithm.
        max_iter (int): Maximum iterations.
        random_state (int): Seed.

    Returns:
        sklearn.pipeline.Pipeline: The linear classification pipeline.
    """
    # Note: Input X is already scaled (metadata) and normalized (embeddings).
    # We pass it directly to Logistic Regression.

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipeline = Pipeline([("classifier", clf)])

    return pipeline
