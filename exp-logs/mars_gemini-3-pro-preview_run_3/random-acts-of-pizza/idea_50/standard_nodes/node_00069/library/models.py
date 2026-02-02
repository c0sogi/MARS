import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import (
    LEXICAL_RF_PARAMS,
    COMMUNITY_RF_PARAMS,
    SEMANTIC_XGB_PARAMS,
    SEMANTIC_RF_PARAMS,
    METADATA_ANCHOR_PARAMS,
    TEMPORAL_LGBM_PARAMS,
    META_LEARNER_PARAMS,
)


class BaseHexModel:
    """
    Base wrapper for Level 1 models in the Hex-View architecture.
    Handles the selection and concatenation of specific feature subsets from the input dictionary.
    Also manages the passing of 'eval_set' for models that support early stopping (XGB/LGBM).
    """

    def __init__(self, model):
        self.model = model

    def _prepare_features(self, X_dict):
        """
        Abstract method to extract and combine features from the dictionary.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _prepare_features")

    def fit(self, X_dict, y, eval_set=None, **kwargs):
        """
        Fits the underlying model.

        Args:
            X_dict (dict): Dictionary containing feature arrays ('lexical', 'community', 'semantic', 'metadata').
            y (array-like): Target labels.
            eval_set (list of tuples, optional): List of (X_val_dict, y_val) for early stopping.
            **kwargs: Additional arguments passed to the underlying fit method.
        """
        X = self._prepare_features(X_dict)

        # Check if the underlying model supports eval_set (Boosters)
        is_booster = isinstance(self.model, (XGBClassifier, LGBMClassifier))

        if eval_set is not None and is_booster:
            # Transform validation sets in eval_set
            # Expected input format: [(X_val_dict, y_val), ...]
            processed_eval_set = []
            for X_val_d, y_val in eval_set:
                processed_eval_set.append((self._prepare_features(X_val_d), y_val))

            self.model.fit(X, y, eval_set=processed_eval_set, **kwargs)
        else:
            # Ignore eval_set for non-boosters (RF, LR) as they don't support it
            self.model.fit(X, y, **kwargs)

        return self

    def predict_proba(self, X_dict):
        """
        Generates probability predictions.
        """
        X = self._prepare_features(X_dict)
        return self.model.predict_proba(X)

    def predict(self, X_dict):
        """
        Generates class predictions.
        """
        X = self._prepare_features(X_dict)
        return self.model.predict(X)


class LexicalBagger(BaseHexModel):
    """
    Lexical Branch: Combines Sparse Lexical TF-IDF with Dense Metadata.
    """

    def _prepare_features(self, X_dict):
        # Use scipy.sparse.hstack to efficiently combine sparse and dense matrices
        return scipy.sparse.hstack([X_dict["lexical"], X_dict["metadata"]])


class CommunityBagger(BaseHexModel):
    """
    Behavioral Branch: Combines Sparse Community TF-IDF with Dense Metadata.
    """

    def _prepare_features(self, X_dict):
        return scipy.sparse.hstack([X_dict["community"], X_dict["metadata"]])


class SemanticBooster(BaseHexModel):
    """
    Semantic Branch (Booster): Combines Dense Embeddings with Dense Metadata.
    """

    def _prepare_features(self, X_dict):
        # Use numpy.hstack for dense-dense combination
        return np.hstack([X_dict["semantic"], X_dict["metadata"]])


class SemanticBagger(BaseHexModel):
    """
    Semantic Branch (Bagger): Combines Dense Embeddings with Dense Metadata.
    """

    def _prepare_features(self, X_dict):
        return np.hstack([X_dict["semantic"], X_dict["metadata"]])


class MetadataAnchor(BaseHexModel):
    """
    Contextual Branch (Linear): Uses Metadata only.
    """

    def _prepare_features(self, X_dict):
        return X_dict["metadata"]


class TemporalBooster(BaseHexModel):
    """
    Contextual Branch (Non-Linear): Uses Metadata only.
    """

    def _prepare_features(self, X_dict):
        return X_dict["metadata"]


class ModelFactory:
    """
    Factory class to instantiate the ensemble components with configured hyperparameters.
    """

    @staticmethod
    def get_level1_models():
        """
        Returns a dictionary of the 6 Level-1 base learners wrapped in their specific classes.
        """
        models = {
            "lexical_bagger": LexicalBagger(
                RandomForestClassifier(**LEXICAL_RF_PARAMS)
            ),
            "community_bagger": CommunityBagger(
                RandomForestClassifier(**COMMUNITY_RF_PARAMS)
            ),
            "semantic_booster": SemanticBooster(XGBClassifier(**SEMANTIC_XGB_PARAMS)),
            "semantic_bagger": SemanticBagger(
                RandomForestClassifier(**SEMANTIC_RF_PARAMS)
            ),
            "metadata_anchor": MetadataAnchor(
                LogisticRegression(**METADATA_ANCHOR_PARAMS)
            ),
            "temporal_booster": TemporalBooster(LGBMClassifier(**TEMPORAL_LGBM_PARAMS)),
        }
        return models

    @staticmethod
    def get_meta_learner():
        """
        Returns the Level-2 Meta-Learner.
        Input to this model is a standard numpy array of probabilities, so no wrapper is needed.
        """
        return LogisticRegression(**META_LEARNER_PARAMS)
