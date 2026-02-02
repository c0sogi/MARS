import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from library.config import XGB_PARAMS, LGBM_PARAMS, RIDGE_PARAMS


def get_xgboost_learner(**kwargs):
    """
    Initializes and returns the XGBoost Regressor configured for GPU acceleration.

    Args:
        **kwargs: Optional arguments to override default configuration.

    Returns:
        xgb.XGBRegressor: The configured XGBoost model.
    """
    # Start with default parameters from config
    params = XGB_PARAMS.copy()

    # Update with any provided kwargs
    params.update(kwargs)

    # Initialize the regressor
    model = xgb.XGBRegressor(**params)

    return model


def get_lightgbm_learner(**kwargs):
    """
    Initializes and returns the LightGBM Regressor configured for CPU execution.

    Args:
        **kwargs: Optional arguments to override default configuration.

    Returns:
        lgb.LGBMRegressor: The configured LightGBM model.
    """
    # Start with default parameters from config
    params = LGBM_PARAMS.copy()

    # Update with any provided kwargs
    params.update(kwargs)

    # Initialize the regressor
    model = lgb.LGBMRegressor(**params)

    return model


def get_meta_learner(**kwargs):
    """
    Initializes and returns the Ridge Regression model used as the meta-learner
    for stacking.

    Args:
        **kwargs: Optional arguments to override default configuration.

    Returns:
        sklearn.linear_model.Ridge: The configured Ridge model.
    """
    # Start with default parameters from config
    params = RIDGE_PARAMS.copy()

    # Update with any provided kwargs
    params.update(kwargs)

    # Initialize the regressor
    model = Ridge(**params)

    return model
