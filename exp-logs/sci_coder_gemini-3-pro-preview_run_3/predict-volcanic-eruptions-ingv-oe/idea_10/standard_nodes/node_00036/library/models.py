import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from library.config import RANDOM_SEED, N_JOBS


def get_base_models(random_seed=RANDOM_SEED, n_jobs=N_JOBS):
    """
    Initializes and returns the dictionary of base learners (Level 0 models)
    for the stacking ensemble.

    Models included:
    1. LightGBM Regressor
    2. XGBoost Regressor
    3. CatBoost Regressor

    Args:
        random_seed (int): Seed for reproducibility.
        n_jobs (int): Number of parallel threads.

    Returns:
        dict: A dictionary where keys are model names ('lgbm', 'xgb', 'cat')
              and values are the initialized model instances.
    """

    # --- LightGBM ---
    # Leaf-wise growth strategy, optimized for MAE
    # Increased n_estimators and decreased learning_rate (Cite Lesson 00002)
    lgbm_model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=31,
        random_state=random_seed,
        verbosity=-1,
        n_jobs=n_jobs,
    )

    # --- XGBoost ---
    # Depth-wise growth, using histogram-based algorithm for speed
    xgb_model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        random_state=random_seed,
        n_jobs=n_jobs,
        tree_method="hist",
        early_stopping_rounds=100,
    )

    # --- CatBoost ---
    # Symmetric trees, robust handling of categorical features (though not used here),
    # and generally strong performance with default-like settings.
    cat_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=5000,
        learning_rate=0.01,
        depth=6,
        random_seed=random_seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=n_jobs,
    )

    return {"lgbm": lgbm_model, "xgb": xgb_model, "cat": cat_model}


def get_meta_model(random_seed=RANDOM_SEED):
    """
    Initializes and returns the meta-learner (Level 1 model).

    Model:
    Ridge Regression - A linear model with L2 regularization to combine
    predictions from base learners without overfitting.

    Args:
        random_seed (int): Seed for reproducibility.

    Returns:
        sklearn.linear_model.Ridge: The initialized meta-learner.
    """
    meta_model = Ridge(alpha=1.0, random_state=random_seed)

    return meta_model
