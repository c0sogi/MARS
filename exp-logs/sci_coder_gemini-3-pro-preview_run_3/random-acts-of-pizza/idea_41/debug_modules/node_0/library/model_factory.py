import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import Config


def get_base_models():
    """
    Instantiates the Level 1 base learners for the Hex-View Stacking Ensemble.

    Returns:
        dict: A dictionary where keys are model names and values are dictionaries containing:
              - 'estimator': The configured model instance.
              - 'feature_sets': List of feature view keys (from feature_engineering) to use.
              - 'sparse': Boolean indicating if the combined features are sparse.
    """
    models = {}

    # 1. Sparse Lexical Branch (Text Modality)
    # Lexical Bagger: RF on TF-IDF + Metadata
    models["lexical_bagger"] = {
        "estimator": RandomForestClassifier(**Config.RF_LEXICAL_PARAMS),
        "feature_sets": ["lexical", "metadata"],
        "sparse": True,
    }

    # 2. Sparse Behavioral Branch (History Modality)
    # Community Bagger: RF on Subreddit History TF-IDF + Metadata
    models["community_bagger"] = {
        "estimator": RandomForestClassifier(**Config.RF_COMMUNITY_PARAMS),
        "feature_sets": ["behavioral", "metadata"],
        "sparse": True,
    }

    # 3. Dense Semantic Branch (Text Modality)
    # Semantic Booster: XGBoost on Embeddings + Metadata
    models["semantic_booster"] = {
        "estimator": XGBClassifier(**Config.XGB_PARAMS),
        "feature_sets": ["semantic", "metadata"],
        "sparse": False,
    }

    # Semantic Gradient: LightGBM on Embeddings + Metadata
    # Provides algorithmic diversity to the dense branch
    models["semantic_gradient"] = {
        "estimator": LGBMClassifier(**Config.LGBM_PARAMS),
        "feature_sets": ["semantic", "metadata"],
        "sparse": False,
    }

    # Semantic Bagger: Random Forest on Embeddings + Metadata
    # Provides structural diversity to the dense branch
    models["semantic_bagger"] = {
        "estimator": RandomForestClassifier(**Config.RF_SEMANTIC_PARAMS),
        "feature_sets": ["semantic", "metadata"],
        "sparse": False,
    }

    # 4. Contextual Branch (Metadata Modality)
    # Metadata Anchor: Logistic Regression on Metadata only
    # Acts as a high-bias regularizer
    models["metadata_anchor"] = {
        "estimator": LogisticRegression(**Config.LR_PARAMS),
        "feature_sets": ["metadata"],
        "sparse": False,
    }

    return models


def get_meta_learner():
    """
    Instantiates the Level 2 Meta-Learner.

    Returns:
        estimator: The configured Logistic Regression model.
    """
    return LogisticRegression(**Config.META_LEARNER_PARAMS)
