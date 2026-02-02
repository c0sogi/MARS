import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from library.config import LGBM_PARAMS, XGB_PARAMS, CATBOOST_PARAMS, RIDGE_PARAMS

# Attempt to import CatBoost as required by the "Idea 12" architecture.
# We wrap this in a try-except block to prevent immediate failure if the
# package is not present in the specific execution environment, although
# the configuration explicitly defines parameters for it.
try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    print(
        "Warning: CatBoost package not found. The CatBoost model will be excluded from the ensemble."
    )
    CATBOOST_AVAILABLE = False


def get_base_models():
    """
    Initializes and returns the dictionary of base learners (Level 0 models)
    for the stacking ensemble.

    This function instantiates the Gradient Boosted Decision Tree models
    defined in the configuration.

    Returns:
        dict: A dictionary where keys are model identifiers ('lgbm', 'xgb', 'catboost')
              and values are the initialized regressor instances.
    """
    models = {}

    # 1. LightGBM Regressor
    # Uses LGBM_PARAMS from config (includes objective='mae', n_estimators=10000, etc.)
    models["lgbm"] = lgb.LGBMRegressor(**LGBM_PARAMS)

    # 2. XGBoost Regressor
    # Uses XGB_PARAMS from config (includes device='cuda', tree_method='hist', etc.)
    models["xgb"] = xgb.XGBRegressor(**XGB_PARAMS)

    # 3. CatBoost Regressor
    # Uses CATBOOST_PARAMS from config (includes task_type='GPU', loss_function='MAE', etc.)
    if CATBOOST_AVAILABLE:
        models["catboost"] = cb.CatBoostRegressor(**CATBOOST_PARAMS)

    return models


def get_meta_learner():
    """
    Initializes and returns the meta-learner (Level 1 model) for the stacking ensemble.

    The meta-learner is a linear model (Ridge Regression) designed to combine
    the Out-of-Fold (OOF) predictions from the base learners.

    Returns:
        sklearn.linear_model.Ridge: The initialized Ridge regression model.
    """
    return Ridge(**RIDGE_PARAMS)
