import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import Ridge
from library.config import Config


def get_base_model(model_name: str, **kwargs):
    """
    Factory function to instantiate base learners with configurations from Config.
    Allows overriding parameters via kwargs for flexibility (e.g., reducing n_estimators for debugging).

    Args:
        model_name (str): The name of the model ('lgbm', 'xgb', 'cat').
        **kwargs: Arbitrary keyword arguments to override default Config parameters.

    Returns:
        model: An instantiated regressor (LGBMRegressor, XGBRegressor, or CatBoostRegressor).
    """
    if model_name == "lgbm":
        params = Config.LGBM_PARAMS.copy()
        params.update(kwargs)
        return lgb.LGBMRegressor(**params)

    elif model_name == "xgb":
        params = Config.XGB_PARAMS.copy()
        params.update(kwargs)
        return xgb.XGBRegressor(**params)

    elif model_name == "cat":
        params = Config.CAT_PARAMS.copy()
        params.update(kwargs)
        return cb.CatBoostRegressor(**params)

    else:
        raise ValueError(
            f"Unknown base model: {model_name}. Expected 'lgbm', 'xgb', or 'cat'."
        )


def get_meta_model(**kwargs):
    """
    Factory function to instantiate the meta-learner (Ridge Regression).
    Allows overriding parameters via kwargs.

    Args:
        **kwargs: Arbitrary keyword arguments to override default Config parameters.

    Returns:
        model: An instantiated Ridge regressor.
    """
    params = {"alpha": Config.RIDGE_ALPHA, "random_state": Config.SEED}
    params.update(kwargs)
    return Ridge(**params)
