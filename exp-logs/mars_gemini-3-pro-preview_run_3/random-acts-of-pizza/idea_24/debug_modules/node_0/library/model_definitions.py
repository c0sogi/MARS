from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from library.config import Config


def get_lexical_bagger(**kwargs):
    """
    Instantiates the Sparse Lexical Branch model (Random Forest).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        RandomForestClassifier: Configured model instance.
    """
    params = Config.MODEL_LEXICAL_RF.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_community_bagger(**kwargs):
    """
    Instantiates the Sparse Behavioral Branch model (Random Forest).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        RandomForestClassifier: Configured model instance.
    """
    params = Config.MODEL_COMMUNITY_RF.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_semantic_booster(**kwargs):
    """
    Instantiates the Dense Semantic Branch model (XGBoost).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        XGBClassifier: Configured model instance.
    """
    params = Config.MODEL_SEMANTIC_XGB.copy()
    params.update(kwargs)
    return XGBClassifier(**params)


def get_semantic_bagger(**kwargs):
    """
    Instantiates the Dense Semantic Branch model (Random Forest).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        RandomForestClassifier: Configured model instance.
    """
    params = Config.MODEL_SEMANTIC_RF.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_manifold_neighbor(**kwargs):
    """
    Instantiates the Manifold Branch model (k-Nearest Neighbors).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        KNeighborsClassifier: Configured model instance.
    """
    params = Config.MODEL_MANIFOLD_KNN.copy()
    params.update(kwargs)
    return KNeighborsClassifier(**params)


def get_metadata_anchor(**kwargs):
    """
    Instantiates the Contextual Branch model (Logistic Regression).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        LogisticRegression: Configured model instance.
    """
    params = Config.MODEL_METADATA_LR.copy()
    params.update(kwargs)
    return LogisticRegression(**params)


def get_meta_learner(**kwargs):
    """
    Instantiates the Level 2 Meta-Learner (Logistic Regression).

    Args:
        **kwargs: Optional overrides for model hyperparameters.

    Returns:
        LogisticRegression: Configured model instance.
    """
    params = Config.MODEL_META_LR.copy()
    params.update(kwargs)
    return LogisticRegression(**params)
