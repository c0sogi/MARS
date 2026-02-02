import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import (
    LOGISTIC_REGRESSION_PARAMS,
    LDA_PARAMS,
)


def get_linear_branch():
    """
    Constructs the Discriminative Linear Branch.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: A Logistic Regression model
        configured with L2 regularization and cross-validation for hyperparameter tuning.
    """
    # Initialize LogisticRegressionCV with parameters from config
    # We unpack the dictionary to pass arguments
    clf = LogisticRegressionCV(**LOGISTIC_REGRESSION_PARAMS)
    return clf


def get_generative_branch():
    """
    Constructs the Generative Linear Branch.

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis: An LDA model
        configured with Ledoit-Wolf shrinkage for robust density estimation.
    """
    # Initialize LDA with parameters from config
    clf = LinearDiscriminantAnalysis(**LDA_PARAMS)
    return clf
