import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import Config


def _create_model(model_class, default_params, **kwargs):
    """
    Helper to instantiate a model with default config params updated by kwargs.
    Ensures the global config dict is not mutated.
    """
    params = copy.deepcopy(default_params)
    params.update(kwargs)
    return model_class(**params)


# =============================================================================
# Branch 1: Sparse Lexical (Text Modality)
# =============================================================================


def get_lexical_bagger(**kwargs):
    """
    Returns the Granular Lexical Bagger (Random Forest).
    Trained on TF-IDF of concatenated Title + Edit-Aware Body.
    """
    return _create_model(RandomForestClassifier, Config.LEXICAL_RF_PARAMS, **kwargs)


# =============================================================================
# Branch 2: Sparse Behavioral (History Modality)
# =============================================================================


def get_community_bagger(**kwargs):
    """
    Returns the Community Bagger (Random Forest).
    Trained on Bag-of-Concepts (Subreddits).
    """
    return _create_model(RandomForestClassifier, Config.COMMUNITY_RF_PARAMS, **kwargs)


# =============================================================================
# Branch 3: Dense Semantic (Text Modality)
# =============================================================================


def get_semantic_booster(**kwargs):
    """
    Returns the Semantic Booster (XGBoost).
    Trained on Dense Embeddings with Conservative Boosting.
    """
    return _create_model(XGBClassifier, Config.SEMANTIC_XGB_PARAMS, **kwargs)


def get_semantic_gradient(**kwargs):
    """
    Returns the Semantic Gradient (LightGBM).
    Trained on Dense Embeddings with Leaf-wise Growth.
    """
    return _create_model(LGBMClassifier, Config.SEMANTIC_LGBM_PARAMS, **kwargs)


def get_semantic_bagger(**kwargs):
    """
    Returns the Semantic Bagger (Random Forest).
    Trained on Dense Embeddings for Structural Diversity.
    """
    return _create_model(RandomForestClassifier, Config.SEMANTIC_RF_PARAMS, **kwargs)


# =============================================================================
# Branch 4: Contextual (Metadata Modality)
# =============================================================================


def get_metadata_anchor(**kwargs):
    """
    Returns the Metadata Anchor (Logistic Regression).
    Acts as a high-bias regularizer on metadata features.
    """
    return _create_model(LogisticRegression, Config.METADATA_LR_PARAMS, **kwargs)


def get_temporal_booster(**kwargs):
    """
    Returns the Temporal Booster (LightGBM).
    Captures non-linear temporal drift in metadata.
    """
    return _create_model(LGBMClassifier, Config.METADATA_LGBM_PARAMS, **kwargs)


# =============================================================================
# Level 2: Meta-Learner
# =============================================================================


def get_meta_learner(**kwargs):
    """
    Returns the Stacking Meta-Learner (Logistic Regression).
    Calibrates the ensemble weights.
    """
    return _create_model(LogisticRegression, Config.META_LEARNER_PARAMS, **kwargs)
