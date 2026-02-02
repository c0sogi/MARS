import copy
from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import Config


def _get_params(
    default_params: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    drop_early_stopping: bool = False,
) -> Dict[str, Any]:
    """
    Helper to merge default parameters with overrides.
    Optionally drops 'early_stopping_rounds' for models that do not accept it in __init__.
    """
    params = copy.deepcopy(default_params)
    if overrides:
        params.update(overrides)

    if drop_early_stopping and "early_stopping_rounds" in params:
        del params["early_stopping_rounds"]

    return params


def get_hept_view_models(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns a dictionary of the 7 Level-1 base learners for the Hept-View architecture.

    Args:
        overrides: Optional dictionary to override parameters for specific models.
                   Format: {'model_name': {'param': value}}

    Returns:
        Dict[str, Any]: Dictionary of initialized model instances.
    """
    models = {}

    # 1. Granular Lexical Bagger (Sparse Text -> RF)
    # Uses TF-IDF of Concatenated Title + Edit-Aware Body + Metadata
    lexical_overrides = overrides.get("lexical_bagger", {}) if overrides else {}
    models["lexical_bagger"] = RandomForestClassifier(
        **_get_params(Config.LEXICAL_BAGGER_PARAMS, lexical_overrides)
    )

    # 2. Community Bagger (Sparse History -> RF)
    # Uses TF-IDF of Subreddit History + Metadata
    community_overrides = overrides.get("community_bagger", {}) if overrides else {}
    models["community_bagger"] = RandomForestClassifier(
        **_get_params(Config.COMMUNITY_BAGGER_PARAMS, community_overrides)
    )

    # 3. Semantic Booster (Dense Text -> XGB)
    # Uses Frozen Compact Embeddings + Metadata
    # XGBoost usually accepts early_stopping_rounds in __init__ in newer versions,
    # but we keep it consistent with Config.
    semantic_boost_overrides = (
        overrides.get("semantic_booster", {}) if overrides else {}
    )
    models["semantic_booster"] = XGBClassifier(
        **_get_params(Config.SEMANTIC_BOOSTER_PARAMS, semantic_boost_overrides)
    )

    # 4. Semantic Gradient (Dense Text -> LGBM)
    # Uses Frozen Compact Embeddings + Metadata
    # LGBMClassifier (sklearn API) typically takes early_stopping_rounds in fit(), not init.
    semantic_grad_overrides = (
        overrides.get("semantic_gradient", {}) if overrides else {}
    )
    models["semantic_gradient"] = LGBMClassifier(
        **_get_params(
            Config.SEMANTIC_GRADIENT_PARAMS,
            semantic_grad_overrides,
            drop_early_stopping=True,
        )
    )

    # 5. Semantic Bagger (Dense Text -> RF)
    # Uses Frozen Compact Embeddings + Metadata
    semantic_bag_overrides = overrides.get("semantic_bagger", {}) if overrides else {}
    models["semantic_bagger"] = RandomForestClassifier(
        **_get_params(Config.SEMANTIC_BAGGER_PARAMS, semantic_bag_overrides)
    )

    # 6. Metadata Anchor (Metadata -> LR)
    # Uses Augmented Global Metadata
    meta_anchor_overrides = overrides.get("metadata_anchor", {}) if overrides else {}
    models["metadata_anchor"] = LogisticRegression(
        **_get_params(Config.METADATA_ANCHOR_PARAMS, meta_anchor_overrides)
    )

    # 7. Temporal Booster (Metadata -> LGBM)
    # Uses Augmented Global Metadata
    # Drop early_stopping_rounds for init safety
    temp_boost_overrides = overrides.get("temporal_booster", {}) if overrides else {}
    models["temporal_booster"] = LGBMClassifier(
        **_get_params(
            Config.TEMPORAL_BOOSTER_PARAMS,
            temp_boost_overrides,
            drop_early_stopping=True,
        )
    )

    return models


def get_meta_learner(overrides: Optional[Dict[str, Any]] = None) -> LogisticRegression:
    """
    Returns the Level-2 Meta-Learner (Logistic Regression).

    Args:
        overrides: Optional dictionary to override parameters.

    Returns:
        LogisticRegression: Initialized meta-learner.
    """
    return LogisticRegression(**_get_params(Config.META_LEARNER_PARAMS, overrides))
