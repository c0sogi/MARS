import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library import config


def get_level1_models(overrides=None):
    """
    Instantiates and returns the dictionary of Level 1 base learners for the stacking ensemble.

    The ensemble consists of 5 models across 3 branches:
    1. Sparse Lexical Branch: Random Forest (Lexical Bagger)
    2. Sparse Behavioral Branch: Random Forest (Community Bagger)
    3. Dense Semantic Branch: XGBoost (Semantic Booster) & Random Forest (Semantic Bagger)
    4. Contextual Branch: Logistic Regression (Metadata Anchor)

    Args:
        overrides (dict, optional): A dictionary of dictionaries to override specific
                                    parameters for models. Keys should match the model keys below.

    Returns:
        dict: A dictionary where keys are model identifiers and values are instantiated model objects.
    """
    if overrides is None:
        overrides = {}

    models = {}

    # 1. Sparse Lexical Branch: Lexical Bagger
    # Random Forest trained on TF-IDF (Title+Body) + Metadata
    lexical_params = config.RF_LEXICAL_PARAMS.copy()
    if "lexical_rf" in overrides:
        lexical_params.update(overrides["lexical_rf"])

    models["lexical_rf"] = RandomForestClassifier(**lexical_params)

    # 2. Sparse Behavioral Branch: Community Bagger
    # Random Forest trained on TF-IDF (Subreddit History) + Metadata
    community_params = config.RF_COMMUNITY_PARAMS.copy()
    if "community_rf" in overrides:
        community_params.update(overrides["community_rf"])

    models["community_rf"] = RandomForestClassifier(**community_params)

    # 3. Dense Semantic Branch: Semantic Booster
    # XGBoost trained on Dense Embeddings + Latent Community Topics + Metadata
    # Note: scale_pos_weight is often handled dynamically in the training loop based on fold balance,
    # but base params are set here.
    xgb_params = config.XGB_SEMANTIC_PARAMS.copy()
    if "semantic_xgb" in overrides:
        xgb_params.update(overrides["semantic_xgb"])

    models["semantic_xgb"] = xgb.XGBClassifier(**xgb_params)

    # 4. Dense Semantic Branch: Semantic Bagger
    # Random Forest trained on Dense Embeddings + Latent Community Topics + Metadata
    # Provides structural diversity to the XGBoost model in the same branch.
    semantic_rf_params = config.RF_SEMANTIC_PARAMS.copy()
    if "semantic_rf" in overrides:
        semantic_rf_params.update(overrides["semantic_rf"])

    models["semantic_rf"] = RandomForestClassifier(**semantic_rf_params)

    # 5. Contextual Branch: Metadata Anchor
    # Logistic Regression trained strictly on Metadata (High-bias regularizer)
    anchor_params = config.LR_ANCHOR_PARAMS.copy()
    if "metadata_lr" in overrides:
        anchor_params.update(overrides["metadata_lr"])

    models["metadata_lr"] = LogisticRegression(**anchor_params)

    return models


def get_meta_learner(overrides=None):
    """
    Instantiates and returns the Level 2 Meta-Learner.

    This model aggregates the probability predictions from the Level 1 base learners.

    Args:
        overrides (dict, optional): Dictionary to override specific parameters.

    Returns:
        sklearn.linear_model.LogisticRegression: The instantiated meta-learner.
    """
    params = config.META_LEARNER_PARAMS.copy()

    if overrides:
        params.update(overrides)

    return LogisticRegression(**params)
