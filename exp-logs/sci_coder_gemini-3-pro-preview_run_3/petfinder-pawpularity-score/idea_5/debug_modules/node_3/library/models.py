import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from lightgbm import LGBMRegressor

from library.config import Config


def get_base_models():
    """
    Returns a dictionary of base (Level 1) models for the stacking ensemble.
    Includes SVR, LightGBM, and Ridge Regression.

    Returns:
        dict: A dictionary where keys are model names ('svr', 'lgbm', 'ridge')
              and values are the instantiated model objects.
    """
    models = {}

    # 1. Support Vector Regression (SVR)
    # Uses RBF kernel. Effective for capturing non-linear relationships in the
    # PCA-compressed feature space.
    # cache_size is increased to 2000MB to speed up fitting if memory allows.
    svr = SVR(kernel="rbf", C=Config.SVR_C, epsilon=Config.SVR_EPSILON, cache_size=2000)
    models["svr"] = svr

    # 2. LightGBM Regressor
    # Gradient boosting model. Parameters are unpacked from Config.
    # Note: verbose is handled within the params in Config (-1 for silent).
    lgbm = LGBMRegressor(**Config.LGBM_PARAMS)
    models["lgbm"] = lgbm

    # 3. Ridge Regression
    # Linear model with L2 regularization. Provides a robust linear baseline.
    # alpha=1.0 is a standard default; random_state ensures reproducibility for solvers that need it.
    ridge = Ridge(alpha=1.0, random_state=Config.SEED)
    models["ridge"] = ridge

    return models


def get_meta_learner():
    """
    Returns the meta-learner (Level 2) model.

    A Linear Meta-Learner aggregates the predictions from the base models.
    We use Ridge Regression with a small alpha (0.1) instead of pure LinearRegression
    to handle potential multicollinearity between the base model predictions
    while maintaining a nearly linear aggregation behavior.

    Returns:
        sklearn.base.BaseEstimator: The instantiated meta-learner model.
    """
    return Ridge(alpha=0.1, random_state=Config.SEED)
