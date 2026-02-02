import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library.config import (
    LEXICAL_BAGGER_PARAMS,
    LEXICAL_ANCHOR_PARAMS,
    COMMUNITY_BAGGER_PARAMS,
    SEMANTIC_BOOSTER_PARAMS,
    SEMANTIC_GRADIENT_PARAMS,
    SEMANTIC_BAGGER_PARAMS,
    METADATA_ANCHOR_PARAMS,
    TEMPORAL_BOOSTER_PARAMS,
    META_LEARNER_PARAMS,
    RANDOM_SEED,
)


def _apply_overrides_and_debug(params, overrides, debug, model_type):
    """
    Helper to merge default params with overrides and apply debug settings.
    """
    # Create a copy to avoid modifying the global config
    final_params = params.copy()

    # Apply manual overrides if provided
    if overrides and isinstance(overrides, dict):
        final_params.update(overrides)

    # Apply debug settings (reduce computational cost)
    if debug:
        if model_type == "rf":
            final_params["n_estimators"] = 10
        elif model_type == "xgb":
            final_params["n_estimators"] = 10
        elif model_type == "lgbm":
            final_params["n_estimators"] = 10
        elif model_type == "linear":
            final_params["max_iter"] = 20

    return final_params


def get_base_models(debug=False, overrides=None):
    """
    Instantiates the 8 base learners for the Oct-View Ensemble.

    Args:
        debug (bool): If True, reduces n_estimators/max_iter for fast debugging.
        overrides (dict, optional): Dictionary of parameter overrides for specific models.
                                    Key should be the model name (e.g., 'lexical_bagger').

    Returns:
        dict: Dictionary mapping model names to instantiated model objects.
    """
    models = {}
    overrides = overrides or {}

    # -------------------------------------------------------------------------
    # 1. LEXICAL BRANCH (Text Modality - Sparse)
    # -------------------------------------------------------------------------

    # Lexical Bagger (Random Forest)
    p_lex_bagger = _apply_overrides_and_debug(
        LEXICAL_BAGGER_PARAMS, overrides.get("lexical_bagger"), debug, "rf"
    )
    models["lexical_bagger"] = RandomForestClassifier(**p_lex_bagger)

    # Lexical Anchor (Logistic Regression)
    p_lex_anchor = _apply_overrides_and_debug(
        LEXICAL_ANCHOR_PARAMS, overrides.get("lexical_anchor"), debug, "linear"
    )
    models["lexical_anchor"] = LogisticRegression(**p_lex_anchor)

    # -------------------------------------------------------------------------
    # 2. BEHAVIORAL BRANCH (History Modality - Sparse)
    # -------------------------------------------------------------------------

    # Community Bagger (Random Forest)
    p_comm_bagger = _apply_overrides_and_debug(
        COMMUNITY_BAGGER_PARAMS, overrides.get("community_bagger"), debug, "rf"
    )
    models["community_bagger"] = RandomForestClassifier(**p_comm_bagger)

    # -------------------------------------------------------------------------
    # 3. SEMANTIC BRANCH (Text Modality - Dense)
    # -------------------------------------------------------------------------

    # Semantic Booster (XGBoost)
    p_sem_booster = _apply_overrides_and_debug(
        SEMANTIC_BOOSTER_PARAMS, overrides.get("semantic_booster"), debug, "xgb"
    )
    models["semantic_booster"] = xgb.XGBClassifier(**p_sem_booster)

    # Semantic Gradient (LightGBM)
    p_sem_gradient = _apply_overrides_and_debug(
        SEMANTIC_GRADIENT_PARAMS, overrides.get("semantic_gradient"), debug, "lgbm"
    )
    models["semantic_gradient"] = lgb.LGBMClassifier(**p_sem_gradient)

    # Semantic Bagger (Random Forest)
    p_sem_bagger = _apply_overrides_and_debug(
        SEMANTIC_BAGGER_PARAMS, overrides.get("semantic_bagger"), debug, "rf"
    )
    models["semantic_bagger"] = RandomForestClassifier(**p_sem_bagger)

    # -------------------------------------------------------------------------
    # 4. CONTEXTUAL BRANCH (Metadata Modality)
    # -------------------------------------------------------------------------

    # Metadata Anchor (Logistic Regression)
    p_meta_anchor = _apply_overrides_and_debug(
        METADATA_ANCHOR_PARAMS, overrides.get("metadata_anchor"), debug, "linear"
    )
    models["metadata_anchor"] = LogisticRegression(**p_meta_anchor)

    # Temporal Booster (LightGBM)
    p_temp_booster = _apply_overrides_and_debug(
        TEMPORAL_BOOSTER_PARAMS, overrides.get("temporal_booster"), debug, "lgbm"
    )
    models["temporal_booster"] = lgb.LGBMClassifier(**p_temp_booster)

    return models


def get_meta_learner(debug=False, overrides=None):
    """
    Instantiates the Level 2 Meta-Learner (Logistic Regression).

    Args:
        debug (bool): If True, reduces max_iter for fast debugging.
        overrides (dict, optional): Dictionary of parameter overrides.

    Returns:
        sklearn.linear_model.LogisticRegression: The meta-learner instance.
    """
    p_meta = _apply_overrides_and_debug(META_LEARNER_PARAMS, overrides, debug, "linear")
    return LogisticRegression(**p_meta)
