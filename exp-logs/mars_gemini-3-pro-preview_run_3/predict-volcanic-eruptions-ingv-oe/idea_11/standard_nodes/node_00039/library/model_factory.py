import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from library.config import Config


def get_stage1_model(n_estimators=None, **kwargs):
    """
    Constructs the Stage 1 Siamese Sensor Encoder model (LightGBM).

    This model is designed to learn generalized seismic patterns from individual
    sensor traces by treating them as independent samples (Siamese architecture).

    Args:
        n_estimators (int, optional): Override for the number of boosting iterations.
                                      Useful for quick debugging or tuning.
        **kwargs: Additional hyperparameter overrides passed to the LGBMRegressor.

    Returns:
        lgb.LGBMRegressor: The configured LightGBM model.
    """
    params = Config.LGBM_PARAMS.copy()

    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    # Apply any additional overrides
    params.update(kwargs)

    return lgb.LGBMRegressor(**params)


def get_stage2_models(n_estimators=None, **kwargs):
    """
    Constructs the dictionary of models for the Stage 2 Spatially-Coupled Stacking.

    This stage uses a heterogeneous ensemble to exploit both the meta-features
    from Stage 1 and the original signal features.

    Args:
        n_estimators (int, optional): Override for the number of estimators/iterations
                                      applied to ALL models in the ensemble.
        **kwargs: Additional hyperparameter overrides applied to ALL models.
                  Note: Ensure kwargs are compatible with LGBM, XGB, and CatBoost.

    Returns:
        dict: A dictionary containing the initialized models with keys 'lgbm', 'xgb', and 'cat'.
    """
    # --- LightGBM ---
    lgb_params = Config.LGBM_PARAMS.copy()
    if n_estimators is not None:
        lgb_params["n_estimators"] = n_estimators
    lgb_params.update(kwargs)

    # --- XGBoost ---
    xgb_params = Config.XGB_PARAMS.copy()
    if n_estimators is not None:
        xgb_params["n_estimators"] = n_estimators
    xgb_params.update(kwargs)

    # --- CatBoost ---
    cat_params = Config.CAT_PARAMS.copy()
    if n_estimators is not None:
        cat_params["iterations"] = n_estimators
    # CatBoost accepts generic kwargs in init which are passed to the parameter dictionary
    cat_params.update(kwargs)

    return {
        "lgbm": lgb.LGBMRegressor(**lgb_params),
        "xgb": xgb.XGBRegressor(**xgb_params),
        "cat": CatBoostRegressor(**cat_params),
    }


def get_stage3_model(alpha=None, **kwargs):
    """
    Constructs the Stage 3 Meta-Learner (Ridge Regression).

    This linear model aggregates the predictions from the Stage 2 ensemble
    to correct systematic biases and minimize variance.

    Args:
        alpha (float, optional): Regularization strength. Defaults to Config.RIDGE_ALPHA.
        **kwargs: Additional arguments passed to the Ridge constructor.

    Returns:
        sklearn.linear_model.Ridge: The configured Ridge regression model.
    """
    ridge_alpha = alpha if alpha is not None else Config.RIDGE_ALPHA

    return Ridge(alpha=ridge_alpha, random_state=Config.SEED, **kwargs)
