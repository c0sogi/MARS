from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


def get_model_registry():
    """
    Returns a dictionary defining the Level 1 Base Learners for the Deca-View Stacking Ensemble.

    Each entry maps a model name to a configuration dictionary containing:
        - 'estimator': The instantiated model object with hyperparameters.
        - 'feature_sets': A list of feature keys (lexical, behavioral, semantic, metadata) to combine.
        - 'is_volatile': Boolean flag indicating if the model requires early stopping/CV-averaging
                         (True) or full-retraining (False).
    """
    registry = {}

    # =========================================================================
    # Branch 1: Sparse Lexical (Text Modality)
    # Input: Concatenated TF-IDF (Title + Body) + Dense Metadata
    # =========================================================================

    # 1. Lexical Bagger (Random Forest)
    registry["lexical_bagger"] = {
        "estimator": RandomForestClassifier(**Config.HP_LEXICAL_BAGGER),
        "feature_sets": ["lexical", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["lexical_bagger"],
    }

    # 2. Lexical Randomizer (ExtraTrees)
    registry["lexical_randomizer"] = {
        "estimator": ExtraTreesClassifier(**Config.HP_LEXICAL_RANDOMIZER),
        "feature_sets": ["lexical", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["lexical_randomizer"],
    }

    # 3. Lexical Anchor (Logistic Regression)
    registry["lexical_anchor"] = {
        "estimator": LogisticRegression(**Config.HP_LEXICAL_ANCHOR),
        "feature_sets": ["lexical", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["lexical_anchor"],
    }

    # =========================================================================
    # Branch 2: Sparse Behavioral (History Modality)
    # Input: TF-IDF (Subreddit History) + Dense Metadata
    # =========================================================================

    # 4. Community Bagger (Random Forest)
    registry["community_bagger"] = {
        "estimator": RandomForestClassifier(**Config.HP_COMMUNITY_BAGGER),
        "feature_sets": ["behavioral", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["community_bagger"],
    }

    # 5. Community Anchor (Logistic Regression)
    registry["community_anchor"] = {
        "estimator": LogisticRegression(**Config.HP_COMMUNITY_ANCHOR),
        "feature_sets": ["behavioral", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["community_anchor"],
    }

    # =========================================================================
    # Branch 3: Dense Semantic (Text Modality)
    # Input: Frozen Embeddings + Dense Metadata
    # =========================================================================

    # 6. Semantic Booster (XGBoost)
    registry["semantic_booster"] = {
        "estimator": XGBClassifier(**Config.HP_SEMANTIC_BOOSTER),
        "feature_sets": ["semantic", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["semantic_booster"],
    }

    # 7. Semantic Gradient (LightGBM)
    registry["semantic_gradient"] = {
        "estimator": LGBMClassifier(**Config.HP_SEMANTIC_GRADIENT),
        "feature_sets": ["semantic", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["semantic_gradient"],
    }

    # 8. Semantic Bagger (Random Forest)
    registry["semantic_bagger"] = {
        "estimator": RandomForestClassifier(**Config.HP_SEMANTIC_BAGGER),
        "feature_sets": ["semantic", "metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["semantic_bagger"],
    }

    # =========================================================================
    # Branch 4: Contextual (Metadata Modality)
    # Input: Dense Metadata Only
    # =========================================================================

    # 9. Metadata Anchor (Logistic Regression)
    registry["metadata_anchor"] = {
        "estimator": LogisticRegression(**Config.HP_METADATA_ANCHOR),
        "feature_sets": ["metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["metadata_anchor"],
    }

    # 10. Temporal Booster (LightGBM)
    registry["temporal_booster"] = {
        "estimator": LGBMClassifier(**Config.HP_TEMPORAL_BOOSTER),
        "feature_sets": ["metadata"],
        "is_volatile": not Config.RETRAIN_FLAGS["temporal_booster"],
    }

    return registry


def get_meta_learner():
    """
    Returns the Level 2 Meta-Learner instance (Logistic Regression) used to stack
    the predictions from the base learners.
    """
    return LogisticRegression(**Config.HP_META_LEARNER)
