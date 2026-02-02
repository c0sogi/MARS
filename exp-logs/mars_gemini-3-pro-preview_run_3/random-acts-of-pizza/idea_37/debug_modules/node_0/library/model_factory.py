import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library import config


def get_base_models():
    """
    Instantiates and returns the dictionary of Level 1 base learners
    configured with hyperparameters from config.py.

    Returns:
        dict: A dictionary where keys are model identifiers and values are
              instantiated sklearn/xgboost model objects.
    """
    models = {}

    # 1. Enhanced Lexical Bagger (Sparse Text Branch)
    # Algorithm: Random Forest
    # Input: Sparse TF-IDF (Title+Body) + Metadata
    models["lexical_bagger"] = RandomForestClassifier(**config.MODEL_LEXICAL_PARAMS)

    # 2. Constrained Community Bagger (Sparse Behavioral Branch)
    # Algorithm: Random Forest
    # Input: Sparse TF-IDF (Subreddits) + Metadata
    models["community_bagger"] = RandomForestClassifier(**config.MODEL_COMMUNITY_PARAMS)

    # 3. Semantic Booster (Dense Text Branch)
    # Algorithm: XGBoost
    # Input: Dense Embeddings + Metadata
    # Note: scale_pos_weight is often updated dynamically in the pipeline based on training data balance
    # We ensure verbosity is set to 0 for silent execution
    xgb_params = config.MODEL_SEMANTIC_XGB_PARAMS.copy()
    if "verbosity" not in xgb_params:
        xgb_params["verbosity"] = 0
    models["semantic_booster"] = xgb.XGBClassifier(**xgb_params)

    # 4. Semantic Bagger (Dense Text Branch)
    # Algorithm: Random Forest
    # Input: Dense Embeddings + Metadata
    models["semantic_bagger"] = RandomForestClassifier(
        **config.MODEL_SEMANTIC_RF_PARAMS
    )

    # 5. Contextual Anchor (Metadata Branch)
    # Algorithm: Logistic Regression
    # Input: Metadata only
    models["contextual_anchor"] = LogisticRegression(**config.MODEL_META_ANCHOR_PARAMS)

    return models


def get_meta_learner():
    """
    Instantiates and returns the Level 2 Stacking Meta-Learner.

    Returns:
        sklearn.linear_model.LogisticRegression: The configured meta-learner.
    """
    return LogisticRegression(**config.STACKING_META_PARAMS)
