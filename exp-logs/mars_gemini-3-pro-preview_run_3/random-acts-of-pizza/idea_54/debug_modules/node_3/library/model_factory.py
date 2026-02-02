from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import (
    LEXICAL_BAGGER_PARAMS,
    LEXICAL_ANCHOR_PARAMS,
    COMMUNITY_BAGGER_PARAMS,
    SEMANTIC_BOOSTER_PARAMS,
    SEMANTIC_GRADIENT_PARAMS,
    SEMANTIC_BAGGER_PARAMS,
    METADATA_ANCHOR_PARAMS,
    TEMPORAL_BOOSTER_PARAMS,
    META_LEARNER_PARAMS,
)


def get_base_learners():
    """
    Returns a dictionary of the 8 initialized Level-1 base learners
    for the Oct-View Stacking Ensemble.
    """
    models = {
        # 1. Sparse Lexical Branch
        "lexical_bagger": RandomForestClassifier(**LEXICAL_BAGGER_PARAMS),
        "lexical_anchor": LogisticRegression(**LEXICAL_ANCHOR_PARAMS),
        # 2. Sparse Behavioral Branch
        "community_bagger": RandomForestClassifier(**COMMUNITY_BAGGER_PARAMS),
        # 3. Dense Semantic Branch
        "semantic_booster": XGBClassifier(**SEMANTIC_BOOSTER_PARAMS),
        "semantic_gradient": LGBMClassifier(**SEMANTIC_GRADIENT_PARAMS),
        "semantic_bagger": RandomForestClassifier(**SEMANTIC_BAGGER_PARAMS),
        # 4. Contextual Branch
        "metadata_anchor": LogisticRegression(**METADATA_ANCHOR_PARAMS),
        "temporal_booster": LGBMClassifier(**TEMPORAL_BOOSTER_PARAMS),
    }

    return models


def get_meta_learner():
    """
    Returns the initialized Level-2 Meta-Learner (Logistic Regression).
    """
    return LogisticRegression(**META_LEARNER_PARAMS)
