import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import log_loss
from library.config import LOGREG_CONFIG, LDA_CONFIG, GPC_CONFIG


def train_logreg_cv(X_train, y_train):
    """
    Trains a Logistic Regression model with Cross-Validation for hyperparameter tuning.
    Optimizes for negative log loss as specified in the config.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: The fitted estimator.
    """
    print("Training Logistic Regression (Discriminative Linear Component)...")

    # Instantiate model with config
    clf = LogisticRegressionCV(**LOGREG_CONFIG)

    # Fit model
    clf.fit(X_train, y_train)

    # Calculate training metric
    y_prob = clf.predict_proba(X_train)
    loss = log_loss(y_train, y_prob)

    print(f"Logistic Regression Training Log Loss: {loss}")

    return clf


def train_lda(X_train, y_train):
    """
    Trains a Linear Discriminant Analysis model with Ledoit-Wolf shrinkage.
    Acts as the Generative Linear Component.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis: The fitted estimator.
    """
    print("Training LDA (Generative Linear Component)...")

    # Instantiate model with config
    clf = LinearDiscriminantAnalysis(**LDA_CONFIG)

    # Fit model
    clf.fit(X_train, y_train)

    # Calculate training metric
    y_prob = clf.predict_proba(X_train)
    loss = log_loss(y_train, y_prob)

    print(f"LDA Training Log Loss: {loss}")

    return clf


def train_gpc(X_train, y_train):
    """
    Trains a Gaussian Process Classifier with an RBF kernel.
    Acts as the Probabilistic Non-Linear Component.

    Args:
        X_train (np.ndarray): Training features (typically PCA-reduced).
        y_train (np.ndarray): Training labels.

    Returns:
        sklearn.gaussian_process.GaussianProcessClassifier: The fitted estimator.
    """
    print("Training GPC (Probabilistic Non-Linear Component)...")

    # Define Kernel (RBF)
    # The length_scale will be optimized during fitting
    kernel = 1.0 * RBF(1.0)

    # Instantiate model with config and kernel
    clf = GaussianProcessClassifier(kernel=kernel, **GPC_CONFIG)

    # Fit model
    clf.fit(X_train, y_train)

    # Calculate training metric
    y_prob = clf.predict_proba(X_train)
    loss = log_loss(y_train, y_prob)

    print(f"GPC Training Log Loss: {loss}")
    print(f"GPC Log Marginal Likelihood: {clf.log_marginal_likelihood_value_}")

    return clf
