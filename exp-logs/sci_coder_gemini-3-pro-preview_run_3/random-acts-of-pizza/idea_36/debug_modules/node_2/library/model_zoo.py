from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library.config import Config


def get_base_models():
    """
    Initializes and returns the dictionary of Level 1 base learners for the
    Pent-View Stacking Ensemble.

    The architecture consists of:
    1. Lexical Bagger (RF): Sparse Text (TF-IDF)
    2. Community Bagger (RF): Sparse History (TF-IDF)
    3. Semantic Booster (XGB): Dense Embeddings + Community Score
    4. Semantic Bagger (RF): Dense Embeddings
    5. Metadata Anchor (LR): Metadata only

    Returns:
        dict: A dictionary mapping model names to initialized sklearn-compatible model objects.
    """
    models = {}

    # 1. Sparse Lexical Branch
    # Random Forest optimized for high-dimensional sparse text data
    models["LexicalBagger"] = RandomForestClassifier(**Config.MODEL_LEXICAL_RF)

    # 2. Sparse Behavioral Branch
    # Random Forest optimized for sparse subreddit history
    models["CommunityBagger"] = RandomForestClassifier(**Config.MODEL_COMMUNITY_RF)

    # 3. Dense Semantic Branch (Boosting)
    # XGBoost optimized for dense embeddings and non-linear interactions with the Community Score
    models["SemanticBooster"] = XGBClassifier(**Config.MODEL_SEMANTIC_XGB)

    # 4. Dense Semantic Branch (Bagging)
    # Random Forest providing structural diversity on dense embeddings
    models["SemanticBagger"] = RandomForestClassifier(**Config.MODEL_SEMANTIC_RF)

    # 5. Contextual Branch
    # Logistic Regression acting as a high-bias regularizer on metadata
    models["MetadataAnchor"] = LogisticRegression(**Config.MODEL_METADATA_LR)

    return models


def get_meta_learner():
    """
    Initializes and returns the Level 2 Meta-Learner.

    This model is responsible for aggregating the probability outputs from the
    base models to form the final prediction.

    Returns:
        sklearn.linear_model.LogisticRegression: The initialized meta-learner.
    """
    return LogisticRegression(**Config.MODEL_META_LEARNER)
