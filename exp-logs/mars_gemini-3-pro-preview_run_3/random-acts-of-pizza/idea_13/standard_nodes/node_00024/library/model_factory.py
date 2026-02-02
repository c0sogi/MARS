import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


def get_lexical_model(**kwargs) -> RandomForestClassifier:
    """
    Instantiates the Lexical-Contextual Bagger model (Random Forest).

    This model operates on the sparse TF-IDF vectors of the request text
    concatenated with the dense global metadata vector.

    Args:
        **kwargs: Arbitrary keyword arguments to override default config parameters.

    Returns:
        RandomForestClassifier: The configured Random Forest model.
    """
    # Start with default parameters from Config
    params = copy.deepcopy(Config.RF_LEXICAL_PARAMS)

    # Update with any overrides provided at runtime
    params.update(kwargs)

    return RandomForestClassifier(**params)


def get_behavioral_model(**kwargs) -> RandomForestClassifier:
    """
    Instantiates the Behavioral-Contextual Bagger model (Random Forest).

    This model operates on the sparse TF-IDF vectors of the user's subreddit history
    concatenated with the dense global metadata vector.

    Args:
        **kwargs: Arbitrary keyword arguments to override default config parameters.

    Returns:
        RandomForestClassifier: The configured Random Forest model.
    """
    # Start with default parameters from Config
    params = copy.deepcopy(Config.RF_BEHAVIORAL_PARAMS)

    # Update with any overrides provided at runtime
    params.update(kwargs)

    return RandomForestClassifier(**params)


def get_semantic_model(**kwargs) -> XGBClassifier:
    """
    Instantiates the Semantic-Contextual Booster model (XGBoost).

    This model operates on the dense SBERT embeddings concatenated with the
    dense global metadata vector. It is configured to use GPU acceleration
    if specified in the config.

    Args:
        **kwargs: Arbitrary keyword arguments to override default config parameters.

    Returns:
        XGBClassifier: The configured XGBoost model.
    """
    # Start with default parameters from Config
    params = copy.deepcopy(Config.XGB_SEMANTIC_PARAMS)

    # Update with any overrides provided at runtime
    params.update(kwargs)

    return XGBClassifier(**params)


def get_meta_learner(**kwargs) -> LogisticRegression:
    """
    Instantiates the Level 2 Meta-Learner (Logistic Regression).

    This model learns to combine the probability outputs from the Level 1 models.

    Args:
        **kwargs: Arbitrary keyword arguments to override default config parameters.

    Returns:
        LogisticRegression: The configured Logistic Regression model.
    """
    # Start with default parameters from Config
    params = copy.deepcopy(Config.META_LEARNER_PARAMS)

    # Update with any overrides provided at runtime
    params.update(kwargs)

    return LogisticRegression(**params)
