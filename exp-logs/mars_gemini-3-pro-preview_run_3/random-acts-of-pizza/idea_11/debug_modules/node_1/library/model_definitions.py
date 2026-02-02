import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


class SparseRandomForest(BaseEstimator, ClassifierMixin):
    """
    A wrapper around RandomForestClassifier optimized for sparse inputs
    (Lexical TF-IDF and Behavioral TF-IDF).
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_PARAMS)

    def fit(self, X, y):
        """
        Fits the Random Forest model.

        Args:
            X (scipy.sparse.csr_matrix): Sparse feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (scipy.sparse.csr_matrix): Sparse feature matrix.

        Returns:
            np.ndarray: Probability of the positive class (class 1).
        """
        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Predicts class labels.
        """
        return self.model.predict(X)


class DenseXGBoost(BaseEstimator, ClassifierMixin):
    """
    A wrapper around XGBClassifier optimized for dense inputs
    (Semantic Embeddings + SVD + Meta features).

    Handles early stopping if validation data is provided.
    """

    def __init__(self):
        # We pass parameters from Config.
        # Note: early_stopping_rounds is in Config.XGB_PARAMS and handled by XGBClassifier init in recent versions.
        self.model = XGBClassifier(**Config.XGB_PARAMS)

    def fit(self, X, y, eval_set=None):
        """
        Fits the XGBoost model.

        Args:
            X (np.ndarray): Dense feature matrix.
            y (np.ndarray): Target labels.
            eval_set (list of tuple): Optional validation set [(X_val, y_val)] for early stopping.
        """
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            # If no eval_set is provided, early stopping (if configured) might warn or disable.
            # We fit without specific eval_set parameters.
            self.model.fit(X, y, verbose=False)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (np.ndarray): Dense feature matrix.

        Returns:
            np.ndarray: Probability of the positive class (class 1).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class StackingMetaLearner(BaseEstimator, ClassifierMixin):
    """
    A Logistic Regression meta-learner that combines probabilities from Level 1 models.
    """

    def __init__(self):
        self.model = LogisticRegression(**Config.META_PARAMS)

    def fit(self, X, y):
        """
        Fits the Meta Learner.

        Args:
            X (np.ndarray): Matrix of probabilities from Level 1 models (N_samples, N_models).
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts final class probabilities.

        Args:
            X (np.ndarray): Matrix of probabilities from Level 1 models.

        Returns:
            np.ndarray: Probability of the positive class (class 1).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)
