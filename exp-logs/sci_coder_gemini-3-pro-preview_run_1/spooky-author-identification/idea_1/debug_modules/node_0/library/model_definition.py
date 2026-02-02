from sklearn.linear_model import LogisticRegression
from library.config import Config


def get_logistic_regression_model(params=None):
    """
    Constructs and returns a Logistic Regression model instance.

    This function initializes a LogisticRegression classifier using the default
    hyperparameters defined in Config.MODEL_PARAMS. It supports optional parameter
    overrides to facilitate hyperparameter tuning or adjustments to training
    iterations (max_iter).

    Args:
        params (dict, optional): A dictionary of hyperparameters to override the
                                 defaults. Example: {'C': 0.1, 'max_iter': 500}.

    Returns:
        sklearn.linear_model.LogisticRegression: An untrained Logistic Regression classifier.
    """
    # Start with the default configuration
    model_params = Config.MODEL_PARAMS.copy()

    # Apply overrides if provided
    if params is not None:
        model_params.update(params)

    # Instantiate the classifier with the combined parameters
    # The dictionary unpacking handles arguments like C, solver, multi_class, random_state, etc.
    model = LogisticRegression(**model_params)

    return model
