import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library.config import (
    RF_ESTIMATORS,
    XGB_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    META_LEARNER_C,
    SEED,
)


class LexicalRF:
    """
    Level 1 Learner: Lexical Bagger (Sparse Text View).
    Uses a Random Forest to capture high-dimensional sparse text signals.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, X, y):
        """
        Fits the Random Forest model.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        return self.model.predict_proba(X)


class BehavioralRF:
    """
    Level 1 Learner: Behavioral Bagger (Sparse History View).
    Uses a Random Forest to capture sparse user subreddit history patterns.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, X, y):
        """
        Fits the Random Forest model.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        return self.model.predict_proba(X)


class SemanticXGB:
    """
    Level 1 Learner: Semantic Booster (Dense Context View).
    Uses XGBoost to capture non-linear interactions in dense embeddings and metadata.
    """

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=XGB_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="auc",
            enable_categorical=False,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fits the XGBoost model.
        Calculates scale_pos_weight dynamically based on training data imbalance.
        Uses early stopping if validation data is provided.
        """
        # Calculate scale_pos_weight dynamically: sum(negative) / sum(positive)
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model.set_params(scale_pos_weight=scale_pos_weight)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        else:
            # Full training without early stopping (e.g. final retrain)
            self.model.fit(X, y, verbose=False)

        return self

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        return self.model.predict_proba(X)


class MetaLearner:
    """
    Level 2 Learner: Stacking Meta-Learner.
    Uses Logistic Regression to calibrate and combine Level 1 predictions.
    """

    def __init__(self):
        self.model = LogisticRegression(
            C=META_LEARNER_C, random_state=SEED, solver="lbfgs", n_jobs=-1
        )

    def fit(self, X, y):
        """
        Fits the Logistic Regression meta-learner.
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        return self.model.predict_proba(X)
