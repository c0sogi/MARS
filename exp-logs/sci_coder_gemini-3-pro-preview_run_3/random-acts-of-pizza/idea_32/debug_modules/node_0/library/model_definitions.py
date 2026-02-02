import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library.config import (
    LEXICAL_RF_PARAMS,
    BEHAVIORAL_RF_PARAMS,
    SEMANTIC_XGB_PARAMS,
    SEMANTIC_RF_PARAMS,
    CONTEXTUAL_LR_PARAMS,
    EARLY_STOPPING_ROUNDS,
    SEED,
)
from library.utils import print_info, print_metric


class BaseLearner:
    """
    Base class for Level 1 learners in the Stacking Ensemble.
    Handles feature concatenation of specific views with the global metadata vector.
    """

    def __init__(self, name, params):
        self.name = name
        self.params = params.copy()
        self.model = None

    def _concat(self, X_view, X_meta):
        """
        Concatenates the specific view (Lexical, Behavioral, Semantic) with Metadata.
        If X_view is None, returns X_meta.
        """
        if X_view is None:
            return X_meta
        if X_meta is None:
            return X_view
        return np.hstack([X_view, X_meta])

    def fit(self, X_view, X_meta, y, X_val_view=None, X_val_meta=None, y_val=None):
        """
        Trains the model. Subclasses can override to handle early stopping.
        """
        X_train = self._concat(X_view, X_meta)
        print_info(f"Training {self.name} with input shape: {X_train.shape}")

        self.model.fit(X_train, y)
        return self

    def predict_proba(self, X_view, X_meta):
        """
        Returns probability of class 1.
        """
        X_test = self._concat(X_view, X_meta)
        return self.model.predict_proba(X_test)[:, 1]


class LexicalBagger(BaseLearner):
    """
    Sparse Lexical Branch: Random Forest on TF-IDF Text + Metadata.
    """

    def __init__(self):
        super().__init__("LexicalBagger", LEXICAL_RF_PARAMS)
        self.model = RandomForestClassifier(**self.params)


class CommunityBagger(BaseLearner):
    """
    Sparse Behavioral Branch: Random Forest on TF-IDF History + Metadata.
    """

    def __init__(self):
        super().__init__("CommunityBagger", BEHAVIORAL_RF_PARAMS)
        self.model = RandomForestClassifier(**self.params)


class SemanticBooster(BaseLearner):
    """
    Dense Semantic Branch: XGBoost on Embeddings + Metadata.
    Implements Early Stopping.
    """

    def __init__(self):
        # Inject early_stopping_rounds into constructor for XGBoost >= 2.0
        params = SEMANTIC_XGB_PARAMS.copy()
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        super().__init__("SemanticBooster", params)
        self.model = XGBClassifier(**self.params)

    def fit(self, X_view, X_meta, y, X_val_view=None, X_val_meta=None, y_val=None):
        X_train = self._concat(X_view, X_meta)
        print_info(f"Training {self.name} with input shape: {X_train.shape}")

        eval_set = None
        if X_val_view is not None and X_val_meta is not None and y_val is not None:
            X_val = self._concat(X_val_view, X_val_meta)
            eval_set = [(X_val, y_val)]
            print_info(f"Early stopping enabled for {self.name}.")

        self.model.fit(X_train, y, eval_set=eval_set, verbose=False)

        if eval_set:
            print_info(f"Best iteration: {self.model.best_iteration}")

        return self


class SemanticBagger(BaseLearner):
    """
    Dense Semantic Branch: Random Forest on Embeddings + Metadata.
    Uses topology-specific regularization (depth constraints).
    """

    def __init__(self):
        super().__init__("SemanticBagger", SEMANTIC_RF_PARAMS)
        self.model = RandomForestClassifier(**self.params)


class MetadataAnchor(BaseLearner):
    """
    Contextual Branch: Logistic Regression on Metadata ONLY.
    Ignores the specific view input.
    """

    def __init__(self):
        super().__init__("MetadataAnchor", CONTEXTUAL_LR_PARAMS)
        self.model = LogisticRegression(**self.params)

    def _concat(self, X_view, X_meta):
        # Ignore X_view, only use X_meta
        return X_meta
