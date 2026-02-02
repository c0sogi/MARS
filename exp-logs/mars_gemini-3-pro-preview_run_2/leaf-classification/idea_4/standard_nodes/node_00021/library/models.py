import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, ClassifierMixin

from library.config import (
    LR_PARAMS,
    LDA_PARAMS,
    RANDOM_SEED,
)


def make_logistic_cv():
    """
    Creates a LogisticRegressionCV instance using configuration from library.config.
    """
    params = LR_PARAMS.copy()
    # Ensure random_state is set
    if "random_state" not in params:
        params["random_state"] = RANDOM_SEED

    return LogisticRegressionCV(**params)


def make_lda():
    """
    Creates a LinearDiscriminantAnalysis instance using configuration from library.config.
    """
    return LinearDiscriminantAnalysis(**LDA_PARAMS)


class HybridEnsemble(BaseEstimator, ClassifierMixin):
    """
    A Soft-Voting Ensemble combining:
    1. Discriminative Linear Model (Logistic Regression)
    2. Generative Linear Model (LDA)

    Removed GPC to prevent ensemble dilution (Cite solution_lesson_node_00009).
    """

    def __init__(self):
        self.lr = make_logistic_cv()
        self.lda = make_lda()
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits all constituent models on the provided dataset.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.

        Returns:
            self
        """
        print("Starting HybridEnsemble training...")

        # 1. Logistic Regression
        print("Fitting Logistic Regression (Discriminative Linear)...")
        self.lr.fit(X, y)
        print("Logistic Regression training complete.")

        # 2. LDA
        print("Fitting LDA (Generative Linear)...")
        self.lda.fit(X, y)
        print("LDA training complete.")

        # Store classes from one of the estimators
        self.classes_ = self.lr.classes_

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using soft voting (averaging).

        Args:
            X (np.ndarray): Features to predict.

        Returns:
            np.ndarray: Averaged probability estimates of shape (n_samples, n_classes).
        """
        # Get probabilities from each component
        # Each returns (n_samples, n_classes)
        probs_lr = self.lr.predict_proba(X)
        probs_lda = self.lda.predict_proba(X)

        # Average the probabilities (Cite solution_lesson_node_00006)
        avg_probs = (probs_lr + probs_lda) / 2.0

        return avg_probs

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Args:
            X (np.ndarray): Features to predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
