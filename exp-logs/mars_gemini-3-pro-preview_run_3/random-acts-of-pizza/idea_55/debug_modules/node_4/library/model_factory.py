import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


def _get_clean_params(params):
    """
    Removes parameters that are not valid for the model constructor
    but might be present in Config for training logic (e.g., early_stopping_rounds).
    """
    clean_params = params.copy()
    # early_stopping_rounds is typically passed to fit(), not __init__
    if "early_stopping_rounds" in clean_params:
        del clean_params["early_stopping_rounds"]
    return clean_params


def get_base_learner(model_name):
    """
    Instantiates a base learner (Level 1) based on the model name.
    """
    if model_name == "lexical_bagger":
        return RandomForestClassifier(**_get_clean_params(Config.LEXICAL_BAGGER_PARAMS))

    elif model_name == "lexical_anchor":
        return LogisticRegression(**_get_clean_params(Config.LEXICAL_ANCHOR_PARAMS))

    elif model_name == "community_bagger":
        return RandomForestClassifier(
            **_get_clean_params(Config.COMMUNITY_BAGGER_PARAMS)
        )

    elif model_name == "community_anchor":
        return LogisticRegression(**_get_clean_params(Config.COMMUNITY_ANCHOR_PARAMS))

    elif model_name == "semantic_booster":
        return XGBClassifier(**_get_clean_params(Config.SEMANTIC_BOOSTER_PARAMS))

    elif model_name == "semantic_gradient":
        return LGBMClassifier(**_get_clean_params(Config.SEMANTIC_GRADIENT_PARAMS))

    elif model_name == "semantic_bagger":
        return RandomForestClassifier(
            **_get_clean_params(Config.SEMANTIC_BAGGER_PARAMS)
        )

    elif model_name == "metadata_anchor":
        return LogisticRegression(**_get_clean_params(Config.METADATA_ANCHOR_PARAMS))

    elif model_name == "temporal_booster":
        return LGBMClassifier(**_get_clean_params(Config.TEMPORAL_BOOSTER_PARAMS))

    else:
        raise ValueError(f"Unknown base learner name: {model_name}")


def get_meta_learner():
    """
    Instantiates the meta learner (Level 2).
    """
    return LogisticRegression(**_get_clean_params(Config.META_LEARNER_PARAMS))


def is_volatile(model_name):
    """
    Determines if a model is volatile (requires CV-Bagging/Early Stopping)
    or stable (can be retrained on full data).
    """
    return model_name in Config.VOLATILE_MODELS


def prepare_model_inputs(features_dict, model_name):
    """
    Selects and concatenates the appropriate feature sets for a given model
    based on the architecture topology.

    Args:
        features_dict (dict): Dictionary containing 'X_lexical', 'X_behavioral',
                              'X_semantic', 'X_metadata' as numpy arrays.
        model_name (str): Name of the model to prepare features for.

    Returns:
        np.ndarray: The concatenated feature matrix for the specific model.
    """
    # All models receive metadata as a baseline
    X_meta = features_dict["X_metadata"]

    # Branch 1: Sparse Lexical (Text + Metadata)
    if model_name in ["lexical_bagger", "lexical_anchor"]:
        # Concatenate Sparse Text TF-IDF with Dense Metadata
        return np.hstack([features_dict["X_lexical"], X_meta])

    # Branch 2: Sparse Behavioral (Community + Metadata)
    elif model_name in ["community_bagger", "community_anchor"]:
        # Concatenate Sparse Community TF-IDF with Dense Metadata
        return np.hstack([features_dict["X_behavioral"], X_meta])

    # Branch 3: Dense Semantic (Embeddings + Metadata)
    elif model_name in ["semantic_booster", "semantic_gradient", "semantic_bagger"]:
        # Concatenate Dense Embeddings with Dense Metadata
        return np.hstack([features_dict["X_semantic"], X_meta])

    # Branch 4: Contextual (Metadata Only)
    elif model_name in ["metadata_anchor", "temporal_booster"]:
        return X_meta

    else:
        raise ValueError(f"Unknown model name for input preparation: {model_name}")
