import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


def get_lexical_bagger():
    """
    Returns the Lexical Bagger (Random Forest) configured with LEXICAL_RF_PARAMS.
    This model is intended to be trained on the Sparse Lexical View (TF-IDF)
    concatenated with the Contextual View (Metadata).
    """
    return RandomForestClassifier(**Config.LEXICAL_RF_PARAMS)


def get_community_bagger():
    """
    Returns the Community Bagger (Random Forest) configured with COMMUNITY_RF_PARAMS.
    This model is intended to be trained on the Sparse Behavioral View (Subreddit History)
    concatenated with the Contextual View (Metadata).
    """
    return RandomForestClassifier(**Config.COMMUNITY_RF_PARAMS)


def get_semantic_booster():
    """
    Returns the Semantic Booster (XGBoost) configured with SEMANTIC_XGB_PARAMS.
    This model is intended to be trained on the Dense Semantic View (Embeddings)
    concatenated with the Contextual View (Metadata).
    """
    # Use deepcopy to prevent side effects if the estimator modifies params internally
    params = copy.deepcopy(Config.SEMANTIC_XGB_PARAMS)
    return XGBClassifier(**params)


def get_semantic_bagger():
    """
    Returns the Semantic Bagger (Random Forest) configured with SEMANTIC_RF_PARAMS.
    This model is intended to be trained on the Dense Semantic View (Embeddings)
    concatenated with the Contextual View (Metadata).
    """
    return RandomForestClassifier(**Config.SEMANTIC_RF_PARAMS)


def get_metadata_anchor():
    """
    Returns the Metadata Anchor (Logistic Regression) configured with METADATA_LOGREG_PARAMS.
    This model is intended to be trained on the Contextual View (Metadata) only.
    """
    return LogisticRegression(**Config.METADATA_LOGREG_PARAMS)


def get_temporal_booster():
    """
    Returns the Temporal Booster (LightGBM) configured with TEMPORAL_LGBM_PARAMS.
    This model is intended to be trained on the Contextual View (Metadata) only,
    specifically leveraging raw timestamps for drift detection.
    """
    params = copy.deepcopy(Config.TEMPORAL_LGBM_PARAMS)
    return LGBMClassifier(**params)


def get_meta_learner():
    """
    Returns the Level 2 Meta-Learner (Logistic Regression) configured with META_LEARNER_PARAMS.
    This model is intended to be trained on the OOF predictions from the base learners.
    """
    return LogisticRegression(**Config.META_LEARNER_PARAMS)


def get_base_learners():
    """
    Utility function to retrieve all base learner factories mapped by their identifiers.

    Returns:
        dict: Mapping of model_name -> factory_function
    """
    return {
        "lexical_bagger": get_lexical_bagger,
        "community_bagger": get_community_bagger,
        "semantic_booster": get_semantic_booster,
        "semantic_bagger": get_semantic_bagger,
        "metadata_anchor": get_metadata_anchor,
        "temporal_booster": get_temporal_booster,
    }
