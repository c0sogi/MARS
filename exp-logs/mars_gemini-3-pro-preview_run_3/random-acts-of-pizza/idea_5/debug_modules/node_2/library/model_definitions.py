import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library.config import (
    LEXICAL_PARAMS,
    SEMANTIC_PARAMS,
    COMMUNITY_PARAMS,
    META_PARAMS,
    SEED,
)
from library.utils import set_seed


class LexicalBagger:
    """
    Level 1 Base Learner: Random Forest on Sparse Lexical Features (TF-IDF) + Metadata.
    """

    def __init__(self):
        set_seed(SEED)
        self.model = RandomForestClassifier(**LEXICAL_PARAMS)
        self.feature_key = "lexical"

    def fit(self, X, y, **kwargs):
        """
        Fits the Random Forest model.

        Args:
            X (dict): Dictionary containing feature matrices. Must contain key 'lexical'.
            y (array-like): Target labels.
            **kwargs: Additional arguments passed to the fit method.
        """
        if self.feature_key not in X:
            raise KeyError(f"Input X must contain key '{self.feature_key}'")

        X_feat = X[self.feature_key]
        self.model.fit(X_feat, y, **kwargs)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (dict): Dictionary containing feature matrices.

        Returns:
            np.ndarray: Probability of the positive class (1).
        """
        if self.feature_key not in X:
            raise KeyError(f"Input X must contain key '{self.feature_key}'")

        X_feat = X[self.feature_key]
        # Return probability of class 1
        return self.model.predict_proba(X_feat)[:, 1]


class SemanticBagger:
    """
    Level 1 Base Learner: Random Forest on Dense Semantic Features (SBERT) + Metadata.
    """

    def __init__(self):
        set_seed(SEED)
        self.model = RandomForestClassifier(**SEMANTIC_PARAMS)
        self.feature_key = "semantic"

    def fit(self, X, y, **kwargs):
        """
        Fits the Random Forest model.

        Args:
            X (dict): Dictionary containing feature matrices. Must contain key 'semantic'.
            y (array-like): Target labels.
            **kwargs: Additional arguments passed to the fit method.
        """
        if self.feature_key not in X:
            raise KeyError(f"Input X must contain key '{self.feature_key}'")

        X_feat = X[self.feature_key]
        self.model.fit(X_feat, y, **kwargs)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (dict): Dictionary containing feature matrices.

        Returns:
            np.ndarray: Probability of the positive class (1).
        """
        if self.feature_key not in X:
            raise KeyError(f"Input X must contain key '{self.feature_key}'")

        X_feat = X[self.feature_key]
        return self.model.predict_proba(X_feat)[:, 1]


class CommunityBooster:
    """
    Level 1 Base Learner: XGBoost on Dense Community Features (Subreddit SVD) + Metadata.
    """

    def __init__(self):
        set_seed(SEED)
        self.model = XGBClassifier(**COMMUNITY_PARAMS)
        self.feature_key = "community"

    def fit(self, X, y, **kwargs):
        """
        Fits the XGBoost model. Supports early stopping if 'eval_set' is provided.

        Args:
            X (dict): Dictionary containing feature matrices. Must contain key 'community'.
            y (array-like): Target labels.
            **kwargs: Additional arguments passed to XGBoost fit (e.g., eval_set, verbose).
        """
        if self.feature_key not in X:
            raise KeyError(f"Input X must contain key '{self.feature_key}'")

        X_feat = X[self.feature_key]

        # Handle eval_set for early stopping
        # If eval_set is provided, it is likely a list of (X_val_dict, y_val) tuples.
        # We need to extract the specific feature matrix from the X_val_dict.
        if "eval_set" in kwargs:
            new_eval_set = []
            for X_val, y_val in kwargs["eval_set"]:
                if isinstance(X_val, dict) and self.feature_key in X_val:
                    new_eval_set.append((X_val[self.feature_key], y_val))
                else:
                    # Fallback if passed as raw matrix or unexpected format
                    new_eval_set.append((X_val, y_val))
            kwargs["eval_set"] = new_eval_set

        self.model.fit(X_feat, y, **kwargs)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (dict): Dictionary containing feature matrices.

        Returns:
            np.ndarray: Probability of the positive class (1).
        """
        if self.feature_key not in X:
            raise KeyError(f"Input X must contain key '{self.feature_key}'")

        X_feat = X[self.feature_key]
        return self.model.predict_proba(X_feat)[:, 1]


class StackingMetaLearner:
    """
    Level 2 Meta Learner: Logistic Regression.
    Combines probabilities from Level 1 models.
    """

    def __init__(self):
        set_seed(SEED)
        self.model = LogisticRegression(**META_PARAMS)

    def fit(self, X, y):
        """
        Fits the meta-learner.

        Args:
            X (np.ndarray): Matrix of shape (n_samples, n_models) containing
                            probabilities from base learners.
            y (array-like): Target labels.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts final class probabilities.

        Args:
            X (np.ndarray): Matrix of shape (n_samples, n_models) containing
                            probabilities from base learners.

        Returns:
            np.ndarray: Probability of the positive class (1).
        """
        return self.model.predict_proba(X)[:, 1]
