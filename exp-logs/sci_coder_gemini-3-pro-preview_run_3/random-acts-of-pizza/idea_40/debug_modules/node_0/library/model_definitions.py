import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import Config


class LexicalBagger:
    """
    Level 1 Model: Random Forest on Sparse Text Features + Dense Metadata.
    """

    def __init__(self):
        self.params = Config.RF_LEXICAL_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_lexical, X_metadata, y):
        """
        Concatenates sparse lexical features and dense metadata, then fits the model.
        """
        # Ensure metadata is 2D
        if X_metadata.ndim == 1:
            X_metadata = X_metadata.reshape(-1, 1)

        # Concatenate sparse matrix and dense array
        X_combined = sp.hstack([X_lexical, X_metadata], format="csr")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_lexical, X_metadata):
        if X_metadata.ndim == 1:
            X_metadata = X_metadata.reshape(-1, 1)
        X_combined = sp.hstack([X_lexical, X_metadata], format="csr")
        return self.model.predict_proba(X_combined)

    def predict(self, X_lexical, X_metadata):
        if X_metadata.ndim == 1:
            X_metadata = X_metadata.reshape(-1, 1)
        X_combined = sp.hstack([X_lexical, X_metadata], format="csr")
        return self.model.predict(X_combined)


class CommunityBagger:
    """
    Level 1 Model: Random Forest on Sparse Behavioral History + Dense Metadata.
    """

    def __init__(self):
        self.params = Config.RF_COMMUNITY_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_community, X_metadata, y):
        """
        Concatenates sparse community features and dense metadata, then fits the model.
        """
        if X_metadata.ndim == 1:
            X_metadata = X_metadata.reshape(-1, 1)

        X_combined = sp.hstack([X_community, X_metadata], format="csr")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_community, X_metadata):
        if X_metadata.ndim == 1:
            X_metadata = X_metadata.reshape(-1, 1)
        X_combined = sp.hstack([X_community, X_metadata], format="csr")
        return self.model.predict_proba(X_combined)

    def predict(self, X_community, X_metadata):
        if X_metadata.ndim == 1:
            X_metadata = X_metadata.reshape(-1, 1)
        X_combined = sp.hstack([X_community, X_metadata], format="csr")
        return self.model.predict(X_combined)


class SemanticBooster:
    """
    Level 1 Model: XGBoost on Dense Semantic Embeddings + Dense Metadata.
    Supports Early Stopping.
    """

    def __init__(self):
        self.params = Config.XGB_SEMANTIC_PARAMS.copy()
        # Extract early stopping rounds to pass to fit()
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", None)
        self.model = XGBClassifier(**self.params)

    def fit(
        self,
        X_semantic,
        X_metadata,
        y,
        X_semantic_val=None,
        X_metadata_val=None,
        y_val=None,
    ):
        """
        Concatenates dense embeddings and metadata.
        If validation data is provided, enables early stopping.
        """
        X_train = np.hstack([X_semantic, X_metadata])

        eval_set = None
        if (
            X_semantic_val is not None
            and X_metadata_val is not None
            and y_val is not None
        ):
            X_val = np.hstack([X_semantic_val, X_metadata_val])
            eval_set = [(X_val, y_val)]

        if eval_set and self.early_stopping_rounds:
            self.model.fit(
                X_train,
                y,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y)
        return self

    def predict_proba(self, X_semantic, X_metadata):
        X_combined = np.hstack([X_semantic, X_metadata])
        return self.model.predict_proba(X_combined)

    def predict(self, X_semantic, X_metadata):
        X_combined = np.hstack([X_semantic, X_metadata])
        return self.model.predict(X_combined)


class SemanticBagger:
    """
    Level 1 Model: Random Forest on Dense Semantic Embeddings + Dense Metadata.
    """

    def __init__(self):
        self.params = Config.RF_SEMANTIC_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_semantic, X_metadata, y):
        X_combined = np.hstack([X_semantic, X_metadata])
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_semantic, X_metadata):
        X_combined = np.hstack([X_semantic, X_metadata])
        return self.model.predict_proba(X_combined)

    def predict(self, X_semantic, X_metadata):
        X_combined = np.hstack([X_semantic, X_metadata])
        return self.model.predict(X_combined)


class InteractionBooster:
    """
    Level 1 Model: LightGBM on Latent Interaction Features (SVD Text + SVD History + Metadata).
    The input X_interaction is assumed to be already concatenated.
    Supports Early Stopping.
    """

    def __init__(self):
        self.params = Config.LGBM_INTERACTION_PARAMS.copy()
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", None)
        self.model = LGBMClassifier(**self.params)

    def fit(self, X_interaction, y, X_interaction_val=None, y_val=None):
        """
        Fits LightGBM on pre-concatenated interaction features.
        """
        eval_set = None
        callbacks = None

        # Setup for early stopping
        if X_interaction_val is not None and y_val is not None:
            eval_set = [(X_interaction_val, y_val)]

        # Note: LightGBM sklearn API handles early_stopping_rounds in fit()
        # but requires eval_set to be present.
        if eval_set and self.early_stopping_rounds:
            from lightgbm import early_stopping, log_evaluation

            callbacks = [
                early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                ),
                log_evaluation(period=0),  # Suppress logging
            ]
            self.model.fit(X_interaction, y, eval_set=eval_set, callbacks=callbacks)
        else:
            self.model.fit(X_interaction, y)
        return self

    def predict_proba(self, X_interaction):
        return self.model.predict_proba(X_interaction)

    def predict(self, X_interaction):
        return self.model.predict(X_interaction)


class MetadataAnchor:
    """
    Level 1 Model: Logistic Regression on Metadata only.
    """

    def __init__(self):
        self.params = Config.LR_ANCHOR_PARAMS.copy()
        self.model = LogisticRegression(**self.params)

    def fit(self, X_metadata, y):
        self.model.fit(X_metadata, y)
        return self

    def predict_proba(self, X_metadata):
        return self.model.predict_proba(X_metadata)

    def predict(self, X_metadata):
        return self.model.predict(X_metadata)


class MetaLearner:
    """
    Level 2 Model: Logistic Regression Stacking Meta-Learner.
    """

    def __init__(self):
        self.params = Config.META_LEARNER_PARAMS.copy()
        self.model = LogisticRegression(**self.params)

    def fit(self, X_oof_predictions, y):
        """
        Fits the meta-learner on Out-Of-Fold predictions from Level 1 models.
        """
        self.model.fit(X_oof_predictions, y)
        return self

    def predict_proba(self, X_test_predictions):
        return self.model.predict_proba(X_test_predictions)

    def predict(self, X_test_predictions):
        return self.model.predict(X_test_predictions)
