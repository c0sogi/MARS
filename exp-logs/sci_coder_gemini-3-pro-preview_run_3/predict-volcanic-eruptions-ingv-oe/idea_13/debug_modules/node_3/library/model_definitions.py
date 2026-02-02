import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from copy import deepcopy
from library.config import LGBM_PARAMS, XGB_PARAMS, HGB_PARAMS, RIDGE_PARAMS


def get_lgbm_regressor(**kwargs):
    """
    Initializes and returns a LightGBM Regressor with parameters defined in config.
    Accepts kwargs to override default parameters.
    """
    params = deepcopy(LGBM_PARAMS)
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def get_xgb_regressor(**kwargs):
    """
    Initializes and returns an XGBoost Regressor with parameters defined in config.
    Accepts kwargs to override default parameters.
    """
    params = deepcopy(XGB_PARAMS)
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def get_catboost_regressor(**kwargs):
    """
    Initializes and returns a HistGradientBoostingRegressor.

    Note: This serves as the replacement for CatBoost as specified in the
    configuration due to environment dependency constraints.
    """
    params = deepcopy(HGB_PARAMS)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def get_meta_learner(**kwargs):
    """
    Initializes and returns the Ridge Regression meta-learner for stacking.
    """
    params = deepcopy(RIDGE_PARAMS)
    params.update(kwargs)
    return Ridge(**params)
