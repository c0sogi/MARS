import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from library.config import (
    L1_RF_LEXICAL_PARAMS,
    L1_RF_BEHAVIORAL_PARAMS,
    L1_XGB_SEMANTIC_PARAMS,
    L1_RF_SEMANTIC_PARAMS,
    L1_LOGREG_CONTEXTUAL_PARAMS,
    L2_META_PARAMS,
)


def get_lexical_rf(**kwargs):
    """
    Returns the Random Forest classifier for the Lexical View (Text TF-IDF + Metadata).

    Args:
        **kwargs: Overrides for default hyperparameters.
    """
    params = L1_RF_LEXICAL_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_behavioral_rf(**kwargs):
    """
    Returns the Random Forest classifier for the Behavioral View (Subreddit History TF-IDF + Metadata).

    Args:
        **kwargs: Overrides for default hyperparameters.
    """
    params = L1_RF_BEHAVIORAL_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_semantic_xgb(**kwargs):
    """
    Returns the XGBoost classifier for the Semantic View (SBERT Embeddings + Metadata).

    Args:
        **kwargs: Overrides for default hyperparameters.
    """
    params = L1_XGB_SEMANTIC_PARAMS.copy()
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def get_semantic_rf(**kwargs):
    """
    Returns the Random Forest classifier for the Semantic View (SBERT Embeddings + Metadata).

    Args:
        **kwargs: Overrides for default hyperparameters.
    """
    params = L1_RF_SEMANTIC_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_contextual_lr(**kwargs):
    """
    Returns the Logistic Regression classifier for the Contextual View (Metadata only).

    Args:
        **kwargs: Overrides for default hyperparameters.
    """
    params = L1_LOGREG_CONTEXTUAL_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)


def get_meta_learner(**kwargs):
    """
    Returns the Level 2 Meta-Learner (Logistic Regression) for Stacking.

    Args:
        **kwargs: Overrides for default hyperparameters.
    """
    params = L2_META_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)
