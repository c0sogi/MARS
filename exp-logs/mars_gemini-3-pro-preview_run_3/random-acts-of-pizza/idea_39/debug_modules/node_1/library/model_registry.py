import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


def get_base_models():
    """
    Instantiates and returns the Level 1 base learners for the Hex-View Stacking Ensemble.

    Returns:
        dict: A dictionary mapping model names to their instantiated sklearn/xgboost objects.
              Keys: 'lexical_bagger', 'community_bagger', 'semantic_booster',
                    'semantic_bagger', 'interaction_booster', 'metadata_anchor'.
    """
    models = {}

    # 1. Lexical Bagger (Sparse Text Branch)
    # Random Forest on TF-IDF vectors
    params_lexical = copy.deepcopy(Config.MODEL_LEXICAL_RF)
    models["lexical_bagger"] = RandomForestClassifier(**params_lexical)

    # 2. Community Bagger (Sparse Behavioral Branch)
    # Random Forest on Subreddit History TF-IDF
    params_community = copy.deepcopy(Config.MODEL_COMMUNITY_RF)
    models["community_bagger"] = RandomForestClassifier(**params_community)

    # 3. Semantic Booster (Dense Semantic Branch)
    # XGBoost on Dense Embeddings
    params_semantic_xgb = copy.deepcopy(Config.MODEL_SEMANTIC_XGB)
    models["semantic_booster"] = XGBClassifier(**params_semantic_xgb)

    # 4. Semantic Bagger (Dense Semantic Branch)
    # Random Forest on Dense Embeddings (Structural Diversity)
    params_semantic_rf = copy.deepcopy(Config.MODEL_SEMANTIC_RF)
    models["semantic_bagger"] = RandomForestClassifier(**params_semantic_rf)

    # 5. Interaction Booster (Low-Rank Interaction Branch)
    # XGBoost on SVD(Text) + SVD(History) + Meta
    params_interaction = copy.deepcopy(Config.MODEL_INTERACTION_XGB)
    models["interaction_booster"] = XGBClassifier(**params_interaction)

    # 6. Metadata Anchor (Contextual Branch)
    # Logistic Regression on Metadata only
    params_metadata = copy.deepcopy(Config.MODEL_METADATA_LR)
    models["metadata_anchor"] = LogisticRegression(**params_metadata)

    return models


def get_meta_model():
    """
    Instantiates and returns the Level 2 Meta-Learner.

    Returns:
        sklearn.linear_model.LogisticRegression: The meta-learner instance.
    """
    # Logistic Regression to calibrate ensemble weights
    params_meta = copy.deepcopy(Config.MODEL_META_LR)
    return LogisticRegression(**params_meta)
