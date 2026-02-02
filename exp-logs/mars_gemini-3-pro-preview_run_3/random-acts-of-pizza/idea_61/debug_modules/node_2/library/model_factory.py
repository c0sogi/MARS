from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import (
    LEXICAL_BAGGER_PARAMS,
    COMMUNITY_BAGGER_PARAMS,
    SEMANTIC_BOOSTER_PARAMS,
    SEMANTIC_GRADIENT_PARAMS,
    SEMANTIC_BAGGER_PARAMS,
    METADATA_ANCHOR_PARAMS,
    TEMPORAL_BOOSTER_PARAMS,
    META_LEARNER_PARAMS,
)


def get_base_models():
    """
    Creates and returns the base learner instances categorized by training stability.

    The 'volatile' group contains Gradient Boosting models (XGBoost, LightGBM) that
    require validation sets for Early Stopping.
    The 'stable' group contains Bagging (Random Forest) and Linear models that are
    trained on the full dataset without Early Stopping.

    Returns:
        dict: A dictionary with keys 'volatile' and 'stable', containing nested
              dictionaries of model instances.
    """

    # --- Volatile Learners (Gradient Boosting) ---

    # 1. Semantic Booster: XGBoost trained on Dense Embeddings
    # Uses conservative learning rate and early stopping
    semantic_booster = XGBClassifier(**SEMANTIC_BOOSTER_PARAMS)

    # 2. Semantic Gradient: LightGBM trained on Dense Embeddings
    # Provides algorithmic diversity via leaf-wise growth
    semantic_gradient = LGBMClassifier(**SEMANTIC_GRADIENT_PARAMS)

    # 3. Temporal Booster: LightGBM trained on Metadata
    # Captures non-linear temporal drift
    temporal_booster = LGBMClassifier(**TEMPORAL_BOOSTER_PARAMS)

    # --- Stable Learners (Bagging & Linear) ---

    # 4. Lexical Bagger: Random Forest trained on Sparse Text TF-IDF
    # Uses granular tokenization to capture specific terms
    lexical_bagger = RandomForestClassifier(**LEXICAL_BAGGER_PARAMS)

    # 5. Community Bagger: Random Forest trained on Sparse Subreddit History
    # Treats history as a Bag-of-Concepts
    community_bagger = RandomForestClassifier(**COMMUNITY_BAGGER_PARAMS)

    # 6. Semantic Bagger: Random Forest trained on Dense Embeddings
    # Uses modality-specific regularization (max_depth)
    semantic_bagger = RandomForestClassifier(**SEMANTIC_BAGGER_PARAMS)

    # 7. Metadata Anchor: Logistic Regression trained on Metadata
    # Acts as a high-bias regularizer
    metadata_anchor = LogisticRegression(**METADATA_ANCHOR_PARAMS)

    return {
        "volatile": {
            "semantic_booster": semantic_booster,
            "semantic_gradient": semantic_gradient,
            "temporal_booster": temporal_booster,
        },
        "stable": {
            "lexical_bagger": lexical_bagger,
            "community_bagger": community_bagger,
            "semantic_bagger": semantic_bagger,
            "metadata_anchor": metadata_anchor,
        },
    }


def get_meta_learner():
    """
    Creates and returns the Level 2 Meta-Learner instance.

    The meta-learner is a Logistic Regression model that calibrates the
    predictions from the seven base learners.

    Returns:
        LogisticRegression: The meta-learner model.
    """
    return LogisticRegression(**META_LEARNER_PARAMS)
