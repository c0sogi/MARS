import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import Timer


class LexicalBagger:
    """
    Level 1 Base Learner: Sparse Lexical Branch.
    Uses RandomForest on concatenated TF-IDF (Title + Body) + Metadata.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_LEXICAL_PARAMS)
        self.name = "LexicalBagger"

    def fit(self, X, y):
        """Standard sklearn fit."""
        with Timer(f"{self.name} Training"):
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns probability of positive class."""
        return self.model.predict_proba(X)[:, 1]


class CommunityBagger:
    """
    Level 1 Base Learner: Sparse Behavioral Branch.
    Uses RandomForest on Subreddit History TF-IDF + Metadata.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_BEHAVIORAL_PARAMS)
        self.name = "CommunityBagger"

    def fit(self, X, y):
        """Standard sklearn fit."""
        with Timer(f"{self.name} Training"):
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns probability of positive class."""
        return self.model.predict_proba(X)[:, 1]


class SemanticBooster:
    """
    Level 1 Base Learner: Dense Semantic Branch (Gradient Boosting).
    Uses XGBoost on Embeddings + Metadata.
    Implements Validation-Guided Early Stopping.
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(**Config.XGB_SEMANTIC_PARAMS)
        self.name = "SemanticBooster"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the XGBoost model.
        If X_val and y_val are provided, uses them for early stopping.
        """
        with Timer(f"{self.name} Training"):
            if X_val is not None and y_val is not None:
                print(
                    f"[{self.name}] Training with Early Stopping (Rounds={Config.XGB_EARLY_STOPPING_ROUNDS})..."
                )
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=Config.XGB_EARLY_STOPPING_ROUNDS,
                    verbose=False,  # Suppress per-tree logs, we print final result manually
                )
                if hasattr(self.model, "best_score"):
                    print(
                        f"[{self.name}] Best Validation Score: {self.model.best_score}"
                    )
            else:
                print(f"[{self.name}] Training on full set (No Early Stopping)...")
                self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Returns probability of positive class."""
        # XGBoost predict_proba returns (N, 2), we want column 1
        return self.model.predict_proba(X)[:, 1]


class SemanticBagger:
    """
    Level 1 Base Learner: Dense Semantic Branch (Bagging).
    Uses RandomForest on Embeddings + Metadata.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_SEMANTIC_PARAMS)
        self.name = "SemanticBagger"

    def fit(self, X, y):
        """Standard sklearn fit."""
        with Timer(f"{self.name} Training"):
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns probability of positive class."""
        return self.model.predict_proba(X)[:, 1]


class MetadataAnchor:
    """
    Level 1 Base Learner: Contextual Branch.
    Uses Logistic Regression on Metadata Only.
    Acts as a high-bias regularizer.
    """

    def __init__(self):
        self.model = LogisticRegression(**Config.LR_CONTEXTUAL_PARAMS)
        self.name = "MetadataAnchor"

    def fit(self, X, y):
        """Standard sklearn fit."""
        with Timer(f"{self.name} Training"):
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns probability of positive class."""
        return self.model.predict_proba(X)[:, 1]


class StackingMetaLearner:
    """
    Level 2 Meta-Learner.
    Combines predictions from Level 1 models using Logistic Regression.
    """

    def __init__(self):
        self.model = LogisticRegression(**Config.META_LEARNER_PARAMS)
        self.name = "StackingMetaLearner"

    def fit(self, X, y):
        """Standard sklearn fit on OOF predictions."""
        with Timer(f"{self.name} Training"):
            self.model.fit(X, y)

        # Print coefficients to see which base models are trusted
        if hasattr(self.model, "coef_"):
            print(f"[{self.name}] Coefficients: {self.model.coef_[0]}")
        return self

    def predict_proba(self, X):
        """Returns probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
