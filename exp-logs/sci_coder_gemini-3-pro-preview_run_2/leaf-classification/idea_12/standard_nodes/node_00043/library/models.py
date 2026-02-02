import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from library.config import (
    LOGISTIC_REGRESSION_PARAMS,
    LDA_PARAMS,
    KERNEL_PCA_PARAMS,
    KERNEL_NYSTROEM_PARAMS,
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


def get_kernel_branch():
    """
    Constructs the Discriminative Kernel Branch.

    Architecture:
        StandardScaler -> PCA -> Nystroem -> LogisticRegressionCV

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn pipeline implementing the
        Nystroem Kernel Logistic Regression approach.
    """
    # 1. Global Scaling
    scaler = StandardScaler()

    # 2. Dimensionality Reduction / Denoising
    pca = PCA(**KERNEL_PCA_PARAMS)

    # 3. Kernel Approximation
    nystroem = Nystroem(**KERNEL_NYSTROEM_PARAMS)

    # 4. Final Linear Classifier
    # Note: We reuse the same LogisticRegressionCV configuration as the linear branch
    # as it effectively solves the linear problem in the projected feature space.
    clf = LogisticRegressionCV(**LOGISTIC_REGRESSION_PARAMS)

    # Construct the pipeline
    pipeline = Pipeline(
        [("scaler", scaler), ("pca", pca), ("nystroem", nystroem), ("classifier", clf)]
    )

    return pipeline
