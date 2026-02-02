import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library import config


def get_level1_model(model_type="rf", **kwargs):
    """
    Factory function to create Level 1 base learners.
    According to the Tri-View Stacking architecture, these are Random Forest Classifiers
    trained on specific views of the data (Lexical, Semantic, Behavioral).

    Args:
        model_type (str): The type of model to create. Defaults to 'rf'.
        **kwargs: Arbitrary keyword arguments to override default configuration defined in config.RF_PARAMS.

    Returns:
        sklearn.base.BaseEstimator: An instance of the requested model (RandomForestClassifier).
    """
    if model_type == "rf":
        # Start with default parameters from configuration
        params = config.RF_PARAMS.copy()

        # Update with any provided keyword arguments
        params.update(kwargs)

        # Instantiate the model
        model = RandomForestClassifier(**params)
        return model
    else:
        raise ValueError(
            f"Unsupported Level 1 model type: {model_type}. Only 'rf' is supported."
        )


def get_meta_model(model_type="lr", **kwargs):
    """
    Factory function to create the Level 2 meta-learner.
    This model aggregates the probability predictions from the Level 1 models.

    Args:
        model_type (str): The type of model to create. Defaults to 'lr'.
        **kwargs: Arbitrary keyword arguments to override default configuration defined in config.LR_PARAMS.

    Returns:
        sklearn.base.BaseEstimator: An instance of the requested model (LogisticRegression).
    """
    if model_type == "lr":
        # Start with default parameters from configuration
        params = config.LR_PARAMS.copy()

        # Update with any provided keyword arguments
        params.update(kwargs)

        # Instantiate the model
        model = LogisticRegression(**params)
        return model
    else:
        raise ValueError(
            f"Unsupported Meta model type: {model_type}. Only 'lr' is supported."
        )
