import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("model_factory")


def get_linear_base_model(**kwargs):
    """
    Instantiates the Base Learner A (Linear Branch).
    This is a Logistic Regression model designed for high-dimensional text embeddings.

    Args:
        **kwargs: Arbitrary keyword arguments to override Config.LR_PARAMS.

    Returns:
        sklearn.linear_model.LogisticRegression: The initialized linear model.
    """
    # Start with default config parameters
    params = Config.LR_PARAMS.copy()

    # Update with any provided overrides
    if kwargs:
        params.update(kwargs)

    logger.info(f"Initializing Linear Base Model with params: {params}")

    model = LogisticRegression(**params)
    return model


def get_tree_base_model(**kwargs):
    """
    Instantiates the Base Learner B (Tree Branch).
    This is a LightGBM model designed for PCA-reduced embeddings and metadata.

    Args:
        **kwargs: Arbitrary keyword arguments to override Config.LGBM_PARAMS.

    Returns:
        lightgbm.LGBMClassifier: The initialized tree model.
    """
    # Start with default config parameters
    params = Config.LGBM_PARAMS.copy()

    # Update with any provided overrides
    if kwargs:
        params.update(kwargs)

    # Ensure verbosity is suppressed if not explicitly set in kwargs (though Config has it)
    if "verbosity" not in params:
        params["verbosity"] = -1

    logger.info(f"Initializing Tree Base Model (LightGBM) with params: {params}")

    model = lgb.LGBMClassifier(**params)
    return model


def get_meta_model(**kwargs):
    """
    Instantiates the Meta-Learner (Level 2).
    This is a Logistic Regression model that combines predictions from base learners.

    Args:
        **kwargs: Arbitrary keyword arguments to override Config.META_LR_PARAMS.

    Returns:
        sklearn.linear_model.LogisticRegression: The initialized meta-model.
    """
    # Start with default config parameters
    params = Config.META_LR_PARAMS.copy()

    # Update with any provided overrides
    if kwargs:
        params.update(kwargs)

    logger.info(f"Initializing Meta-Model with params: {params}")

    model = LogisticRegression(**params)
    return model
