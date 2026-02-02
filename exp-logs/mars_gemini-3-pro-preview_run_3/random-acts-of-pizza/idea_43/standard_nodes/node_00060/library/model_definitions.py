import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


def get_base_learner(learner_name, **kwargs):
    """
    Factory function to instantiate Level 1 base learners for the Hex-View Stacking Ensemble.

    Args:
        learner_name (str): The name of the learner to instantiate.
                            Options: 'lexical_bagger', 'community_bagger',
                                     'semantic_booster', 'semantic_gradient',
                                     'semantic_bagger', 'metadata_anchor'.
        **kwargs: Additional keyword arguments to override default configuration parameters.

    Returns:
        sklearn-compatible estimator: The initialized model instance.
    """
    # 1. Sparse Lexical Branch (Random Forest)
    if learner_name == "lexical_bagger":
        params = Config.LEXICAL_RF_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    # 2. Sparse Behavioral Branch (Random Forest)
    elif learner_name == "community_bagger":
        params = Config.BEHAVIORAL_RF_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    # 3. Semantic Booster (XGBoost)
    elif learner_name == "semantic_booster":
        params = Config.XGB_PARAMS.copy()
        params.update(kwargs)
        # XGBClassifier from xgboost package
        return XGBClassifier(**params)

    # 4. Semantic Gradient (LightGBM)
    elif learner_name == "semantic_gradient":
        params = Config.LGBM_PARAMS.copy()
        params.update(kwargs)
        # LGBMClassifier from lightgbm package
        return LGBMClassifier(**params)

    # 5. Semantic Bagger (Random Forest)
    elif learner_name == "semantic_bagger":
        params = Config.SEMANTIC_RF_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    # 6. Metadata Anchor (Logistic Regression)
    elif learner_name == "metadata_anchor":
        params = Config.METADATA_LR_PARAMS.copy()
        params.update(kwargs)
        return LogisticRegression(**params)

    else:
        valid_names = [
            "lexical_bagger",
            "community_bagger",
            "semantic_booster",
            "semantic_gradient",
            "semantic_bagger",
            "metadata_anchor",
        ]
        raise ValueError(
            f"Unknown learner_name: '{learner_name}'. "
            f"Valid options are: {valid_names}"
        )


def get_meta_learner(**kwargs):
    """
    Factory function to instantiate the Level 2 Meta-Learner.

    Args:
        **kwargs: Additional keyword arguments to override default configuration parameters.

    Returns:
        sklearn.linear_model.LogisticRegression: The initialized meta-learner.
    """
    params = Config.META_LR_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)
