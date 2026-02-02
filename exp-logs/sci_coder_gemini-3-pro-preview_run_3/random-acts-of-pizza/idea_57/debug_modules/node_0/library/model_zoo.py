import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger("model_zoo")


def get_model(model_key, **kwargs):
    """
    Factory function to instantiate models based on the configuration key.

    Args:
        model_key (str): The key identifying the model in Config.MODEL_CONFIGS
                         or 'meta_learner'.
        **kwargs: Additional arguments to override default hyperparameters.

    Returns:
        object: An instantiated scikit-learn, XGBoost, or LightGBM model object.
    """

    # 1. Handle Meta-Learner (Level 2)
    if model_key == "meta_learner":
        logger.info("Initializing Meta-Learner (Logistic Regression)...")
        params = Config.META_LEARNER_PARAMS.copy()
        params.update(kwargs)
        return LogisticRegression(**params)

    # 2. Handle Base Learners (Level 1)
    if model_key not in Config.MODEL_CONFIGS:
        raise ValueError(f"Model key '{model_key}' not found in Config.MODEL_CONFIGS.")

    model_conf = Config.MODEL_CONFIGS[model_key]
    model_type = model_conf["type"]

    # Copy params to avoid mutating the global config
    params = model_conf["params"].copy()

    # Update with any runtime overrides
    params.update(kwargs)

    logger.info(f"Initializing {model_key} (Type: {model_type})...")

    # 3. Instantiate based on type
    if model_type == "sklearn_rf":
        return RandomForestClassifier(**params)

    elif model_type == "sklearn_lr":
        return LogisticRegression(**params)

    elif model_type == "xgboost":
        # Ensure n_jobs is set if not already (though Config usually has it)
        if "n_jobs" not in params:
            params["n_jobs"] = Config.N_JOBS
        return XGBClassifier(**params)

    elif model_type == "lightgbm":
        # Ensure n_jobs is set
        if "n_jobs" not in params:
            params["n_jobs"] = Config.N_JOBS
        return LGBMClassifier(**params)

    else:
        raise ValueError(f"Unknown model type '{model_type}' for key '{model_key}'.")
