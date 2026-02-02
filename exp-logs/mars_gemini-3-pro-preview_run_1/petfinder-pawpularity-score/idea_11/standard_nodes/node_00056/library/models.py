from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from library.config import Config


def get_ridge_expert():
    """
    Returns a Ridge Regression expert with built-in Cross-Validation for alpha selection.

    The Linear Expert maps global linear trends in the feature space.
    RidgeCV efficiently performs Leave-One-Out Cross-Validation to select the optimal
    regularization strength (alpha) from the list defined in Config.
    """
    return RidgeCV(alphas=Config.RIDGE_ALPHAS, scoring="neg_root_mean_squared_error")


def get_svr_expert():
    """
    Returns a Support Vector Regression expert wrapped in GridSearchCV.

    The Kernel Expert captures non-linear manifolds based on embedding distance.
    Since SVR hyperparameters (C, epsilon) are critical and data-dependent,
    this function returns a GridSearchCV object that will automatically find the
    best configuration from Config.SVR_PARAMS during fitting.
    """
    # Extract parameters from Config
    svr_params = Config.SVR_PARAMS.copy()

    # Separate fixed parameters from the grid search parameters
    # The grid search params are lists in the config
    param_grid = {"C": svr_params.pop("C"), "epsilon": svr_params.pop("epsilon")}

    # The remaining params (kernel, cache_size) are fixed arguments for the estimator
    base_svr = SVR(**svr_params)

    # Initialize GridSearchCV
    # We use the global N_FOLDS for the internal cross-validation of the expert
    return GridSearchCV(
        estimator=base_svr,
        param_grid=param_grid,
        cv=Config.N_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )


def get_extratrees_expert():
    """
    Returns an Extra Trees Regressor expert.

    The Bagging Expert captures conditional logic via randomized decision trees.
    It uses the robust hyperparameters defined in Config.ET_PARAMS.
    """
    return ExtraTreesRegressor(**Config.ET_PARAMS)


def get_lgbm_expert(override_params=None):
    """
    Returns a LightGBM Regressor expert.

    The Boosting Expert reduces bias and learns subtle sequential interactions.
    It uses the parameters defined in Config.LGBM_PARAMS, which can be optionally
    overridden.

    Args:
        override_params (dict, optional): Dictionary of parameters to override defaults.
    """
    params = Config.get_lgbm_params(override_params)
    return LGBMRegressor(**params)


def get_meta_learner():
    """
    Returns the Level-1 Meta-Learner (Bayesian Ridge Regressor).

    The Meta-Learner aggregates predictions from all Level-0 experts.
    Bayesian Ridge is chosen for its ability to handle correlated inputs (expert predictions)
    and provide a robust probabilistic estimate.
    """
    return BayesianRidge(**Config.META_MODEL_PARAMS)
