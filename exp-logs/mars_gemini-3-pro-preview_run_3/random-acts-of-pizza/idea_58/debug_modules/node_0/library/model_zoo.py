import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


def get_lexical_bagger(**kwargs):
    """
    Instantiates the Granular Lexical Bagger (Random Forest) for the Sparse Lexical Branch.
    Uses LEXICAL_BAGGER_PARAMS from Config.
    """
    params = Config.LEXICAL_BAGGER_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_community_bagger(**kwargs):
    """
    Instantiates the Community Bagger (Random Forest) for the Sparse Behavioral Branch.
    Uses COMMUNITY_BAGGER_PARAMS from Config.
    """
    params = Config.COMMUNITY_BAGGER_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_semantic_booster(**kwargs):
    """
    Instantiates the Semantic Booster (XGBoost) for the Dense Semantic Branch.
    Uses SEMANTIC_BOOSTER_PARAMS from Config.
    """
    params = Config.SEMANTIC_BOOSTER_PARAMS.copy()
    params.update(kwargs)
    # XGBClassifier accepts kwargs like early_stopping_rounds, though they are often used in fit().
    # Passing them here ensures they are stored in the estimator instance.
    return XGBClassifier(**params)


def get_semantic_gradient(**kwargs):
    """
    Instantiates the Semantic Gradient (LightGBM) for the Dense Semantic Branch.
    Uses SEMANTIC_GRADIENT_PARAMS from Config.
    """
    params = Config.SEMANTIC_GRADIENT_PARAMS.copy()
    params.update(kwargs)
    return LGBMClassifier(**params)


def get_semantic_bagger(**kwargs):
    """
    Instantiates the Semantic Bagger (Random Forest) for the Dense Semantic Branch.
    Uses SEMANTIC_BAGGER_PARAMS from Config.
    """
    params = Config.SEMANTIC_BAGGER_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_metadata_anchor(**kwargs):
    """
    Instantiates the Metadata Anchor (Logistic Regression) for the Contextual Branch.
    Uses METADATA_ANCHOR_PARAMS from Config.
    """
    params = Config.METADATA_ANCHOR_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)


def get_temporal_booster(**kwargs):
    """
    Instantiates the Temporal Booster (LightGBM) for the Contextual Branch.
    Uses TEMPORAL_BOOSTER_PARAMS from Config.
    """
    params = Config.TEMPORAL_BOOSTER_PARAMS.copy()
    params.update(kwargs)
    return LGBMClassifier(**params)


def get_meta_learner(**kwargs):
    """
    Instantiates the Level 2 Meta-Learner (Logistic Regression).
    Uses META_LEARNER_PARAMS from Config.
    """
    params = Config.META_LEARNER_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)
