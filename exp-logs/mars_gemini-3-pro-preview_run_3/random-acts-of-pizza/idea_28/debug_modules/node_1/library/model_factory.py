import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config

# Configure logging (optional but good practice)
logger = logging.getLogger(__name__)


def get_lexical_bagger() -> RandomForestClassifier:
    """
    Creates the Lexical Bagger model (Random Forest).

    Intended Input: Sparse TF-IDF matrix of request text + Global Metadata.
    Configuration: Uses Config.RF_PARAMS (includes min_samples_leaf=2 for regularization).
    """
    return RandomForestClassifier(**Config.RF_PARAMS)


def get_community_bagger() -> RandomForestClassifier:
    """
    Creates the Community Bagger model (Random Forest).

    Intended Input: Sparse TF-IDF matrix of subreddit history + Global Metadata.
    Configuration: Uses Config.RF_PARAMS.
    """
    return RandomForestClassifier(**Config.RF_PARAMS)


def get_semantic_booster() -> XGBClassifier:
    """
    Creates the Semantic Booster model (XGBoost).

    Intended Input: Dense Embeddings of request text + Global Metadata.
    Configuration: Uses Config.XGB_PARAMS (configured for GPU/CUDA).

    Note: Early stopping parameters (eval_set, early_stopping_rounds) should be
    passed to the .fit() method during training, not here.
    """
    return XGBClassifier(**Config.XGB_PARAMS)


def get_semantic_bagger() -> RandomForestClassifier:
    """
    Creates the Semantic Bagger model (Random Forest).

    Intended Input: Dense Embeddings of request text + Global Metadata.
    Configuration: Uses Config.RF_PARAMS.
    """
    return RandomForestClassifier(**Config.RF_PARAMS)


def get_metadata_anchor() -> LogisticRegression:
    """
    Creates the Metadata Anchor model (Logistic Regression).

    Intended Input: Global Metadata Vector (Scaled).
    Configuration: Uses Config.LR_PARAMS (High bias regularizer).
    """
    return LogisticRegression(**Config.LR_PARAMS)


def get_meta_learner() -> LogisticRegression:
    """
    Creates the Level 2 Meta-Learner (Logistic Regression).

    Intended Input: Probabilities from Level 1 base learners.
    Configuration: Uses Config.LR_PARAMS to calibrate ensemble weights.
    """
    return LogisticRegression(**Config.LR_PARAMS)
