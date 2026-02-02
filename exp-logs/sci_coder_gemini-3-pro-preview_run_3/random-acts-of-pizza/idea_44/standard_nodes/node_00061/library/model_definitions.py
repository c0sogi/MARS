import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from library import config
from library import utils


class LexicalBagger:
    """
    Level 1 Base Learner: Sparse Lexical Branch.
    Uses Random Forest on concatenated TF-IDF vectors (Title + Body) + Global Metadata.
    """

    def __init__(self):
        self.params = config.RF_LEXICAL_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)
        self.logger = utils.get_logger("LexicalBagger")

    def fit(self, X_lexical, X_meta, y, eval_set=None):
        """
        Fits the model.
        Args:
            X_lexical: Sparse TF-IDF matrix.
            X_meta: Dense metadata matrix.
            y: Target labels.
            eval_set: Tuple (X_val_lexical, X_val_meta, y_val) for validation (not used by RF but kept for signature).
        """
        # Combine sparse lexical features with dense metadata
        X_combined = sparse.hstack((X_lexical, X_meta)).tocsr()
        self.logger.info(f"Training LexicalBagger on shape {X_combined.shape}")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_lexical, X_meta):
        X_combined = sparse.hstack((X_lexical, X_meta)).tocsr()
        return self.model.predict_proba(X_combined)


class CommunityBagger:
    """
    Level 1 Base Learner: Sparse Behavioral Branch.
    Uses Random Forest on Subreddit History TF-IDF + Global Metadata.
    """

    def __init__(self):
        self.params = config.RF_COMMUNITY_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)
        self.logger = utils.get_logger("CommunityBagger")

    def fit(self, X_community, X_meta, y, eval_set=None):
        """
        Fits the model.
        Args:
            X_community: Sparse TF-IDF matrix (subreddits).
            X_meta: Dense metadata matrix.
            y: Target labels.
            eval_set: Tuple (X_val_comm, X_val_meta, y_val) (not used by RF).
        """
        X_combined = sparse.hstack((X_community, X_meta)).tocsr()
        self.logger.info(f"Training CommunityBagger on shape {X_combined.shape}")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_community, X_meta):
        X_combined = sparse.hstack((X_community, X_meta)).tocsr()
        return self.model.predict_proba(X_combined)


class SemanticBooster:
    """
    Level 1 Base Learner: Dense Semantic Branch (Boosting).
    Uses XGBoost on Dense Embeddings + Global Metadata.
    Supports Early Stopping.
    """

    def __init__(self):
        self.params = config.XGB_SEMANTIC_PARAMS.copy()
        # Extract early_stopping_rounds if present in params to pass to fit/constructor
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)
        self.model = XGBClassifier(**self.params)
        self.logger = utils.get_logger("SemanticBooster")

    def fit(self, X_semantic, X_meta, y, eval_set=None):
        """
        Fits the model.
        Args:
            X_semantic: Dense embedding matrix.
            X_meta: Dense metadata matrix.
            y: Target labels.
            eval_set: Tuple (X_val_sem, X_val_meta, y_val) for early stopping.
        """
        X_combined = np.hstack((X_semantic, X_meta))

        fit_params = {"verbose": False}

        if eval_set:
            X_val_sem, X_val_meta, y_val = eval_set
            X_val_combined = np.hstack((X_val_sem, X_val_meta))

            fit_params["eval_set"] = [(X_combined, y), (X_val_combined, y_val)]

            # XGBoost >= 1.6 requires early_stopping_rounds in init or set_params
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)

            self.logger.info(
                f"Training SemanticBooster with Early Stopping (rounds={self.early_stopping_rounds})"
            )
        else:
            # Disable early stopping requirement when no eval set is provided
            self.model.set_params(early_stopping_rounds=None)
            self.logger.info("Training SemanticBooster on full set (No Early Stopping)")

        self.model.fit(X_combined, y, **fit_params)
        return self

    def predict_proba(self, X_semantic, X_meta):
        X_combined = np.hstack((X_semantic, X_meta))
        return self.model.predict_proba(X_combined)


class SemanticBagger:
    """
    Level 1 Base Learner: Dense Semantic Branch (Bagging).
    Uses Random Forest on Dense Embeddings + Global Metadata.
    """

    def __init__(self):
        self.params = config.RF_SEMANTIC_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)
        self.logger = utils.get_logger("SemanticBagger")

    def fit(self, X_semantic, X_meta, y, eval_set=None):
        X_combined = np.hstack((X_semantic, X_meta))
        self.logger.info(f"Training SemanticBagger on shape {X_combined.shape}")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_semantic, X_meta):
        X_combined = np.hstack((X_semantic, X_meta))
        return self.model.predict_proba(X_combined)


class MetadataAnchor:
    """
    Level 1 Base Learner: Contextual Branch (Linear).
    Uses Logistic Regression on Global Metadata only.
    Acts as a high-bias regularizer.
    """

    def __init__(self):
        self.params = config.LR_METADATA_PARAMS.copy()
        self.model = LogisticRegression(**self.params)
        self.logger = utils.get_logger("MetadataAnchor")

    def fit(self, X_ignored, X_meta, y, eval_set=None):
        """
        Fits the model.
        Args:
            X_ignored: Ignored (placeholder for consistency).
            X_meta: Dense metadata matrix.
            y: Target labels.
        """
        self.logger.info(f"Training MetadataAnchor on shape {X_meta.shape}")
        self.model.fit(X_meta, y)
        return self

    def predict_proba(self, X_ignored, X_meta):
        return self.model.predict_proba(X_meta)


class TemporalBooster:
    """
    Level 1 Base Learner: Contextual Branch (Non-Linear).
    Uses LightGBM on Global Metadata only.
    Captures temporal regimes and non-linear user stat interactions.
    """

    def __init__(self):
        self.params = config.LGBM_TEMPORAL_PARAMS.copy()
        self.model = LGBMClassifier(**self.params)
        self.logger = utils.get_logger("TemporalBooster")

    def fit(self, X_ignored, X_meta, y, eval_set=None):
        """
        Fits the model.
        Args:
            X_ignored: Ignored.
            X_meta: Dense metadata matrix.
            y: Target labels.
            eval_set: Tuple (X_val_ignored, X_val_meta, y_val) for early stopping.
        """
        callbacks = []

        if eval_set:
            _, X_val_meta, y_val = eval_set
            eval_s = [(X_meta, y), (X_val_meta, y_val)]

            # Add early stopping callback
            # Note: LightGBM python-package uses callbacks for early stopping
            callbacks.append(early_stopping(stopping_rounds=50, verbose=False))
            callbacks.append(log_evaluation(period=0))  # Suppress verbose logging

            self.logger.info("Training TemporalBooster with Early Stopping")
            self.model.fit(X_meta, y, eval_set=eval_s, callbacks=callbacks)
        else:
            self.logger.info("Training TemporalBooster on full set")
            self.model.fit(X_meta, y)

        return self

    def predict_proba(self, X_ignored, X_meta):
        return self.model.predict_proba(X_meta)
