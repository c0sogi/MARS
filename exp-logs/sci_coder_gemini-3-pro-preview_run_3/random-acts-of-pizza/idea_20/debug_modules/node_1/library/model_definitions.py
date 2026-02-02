from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from library.config import RF_PARAMS, XGB_PARAMS, KNN_PARAMS, LR_PARAMS


def get_lexical_bagger(**kwargs):
    """
    Returns the Random Forest model for the Sparse Lexical Branch.
    Uses TF-IDF text features.
    """
    params = RF_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_community_bagger(**kwargs):
    """
    Returns the Random Forest model for the Sparse Behavioral Branch.
    Uses TF-IDF subreddit history features.
    """
    params = RF_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_semantic_booster(**kwargs):
    """
    Returns the XGBoost model for the Dense Unified Branch.
    Uses dense embeddings and metadata.
    """
    params = XGB_PARAMS.copy()
    params.update(kwargs)
    return XGBClassifier(**params)


def get_semantic_bagger(**kwargs):
    """
    Returns the Random Forest model for the Dense Unified Branch.
    Provides structural stability alongside the booster.
    """
    params = RF_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_manifold_neighbor(**kwargs):
    """
    Returns the k-Nearest Neighbors model for the Dense Unified Branch.
    Uses Cosine similarity to exploit embedding manifold structure.
    """
    params = KNN_PARAMS.copy()
    params.update(kwargs)
    return KNeighborsClassifier(**params)


def get_metadata_anchor(**kwargs):
    """
    Returns the Logistic Regression model for the Contextual Branch.
    Acts as a high-bias regularizer on metadata features.
    """
    params = LR_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)


def get_meta_learner(**kwargs):
    """
    Returns the Logistic Regression model for the Level 2 Stacking layer.
    Calibrates the ensemble weights.
    """
    params = LR_PARAMS.copy()
    params.update(kwargs)
    return LogisticRegression(**params)
