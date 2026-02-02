import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library.config import (
    LEXICAL_BAGGER_PARAMS,
    COMMUNITY_BAGGER_PARAMS,
    SEMANTIC_BOOSTER_PARAMS,
    SEMANTIC_GRADIENT_PARAMS,
    SEMANTIC_BAGGER_PARAMS,
    METADATA_ANCHOR_PARAMS,
    TEMPORAL_BOOSTER_PARAMS,
    META_LEARNER_PARAMS,
    SEED,
)


class FeatureBinder(BaseEstimator, ClassifierMixin):
    """
    A wrapper that binds specific feature subsets from a dictionary of features
    to an underlying estimator. This implements the 'Topology Matching'
    aspect of the architecture.
    """

    def __init__(self, estimator, feature_keys):
        self.estimator = estimator
        self.feature_keys = feature_keys

    def _prepare_X(self, X_dict):
        """
        Extracts and concatenates the required features from the input dictionary.
        Handles both sparse and dense matrices.
        """
        # If input is not a dict, assume it's already processed (e.g. for MetaLearner)
        if not isinstance(X_dict, dict):
            return X_dict

        if self.feature_keys is None:
            raise ValueError("Feature keys must be defined for dictionary input.")

        components = []
        for key in self.feature_keys:
            if key not in X_dict:
                raise KeyError(f"Feature key '{key}' not found in input dictionary.")
            components.append(X_dict[key])

        if not components:
            raise ValueError("No features selected.")

        # Check if any component is sparse
        is_sparse = any(sp.issparse(c) for c in components)

        if is_sparse:
            # Use scipy.sparse.hstack (handles mixed sparse/dense automatically)
            # It converts dense arrays to sparse CSR before stacking if one input is sparse
            return sp.hstack(components, format="csr")
        else:
            # Use numpy.hstack for purely dense arrays
            return np.hstack(components)

    def fit(self, X, y, **kwargs):
        """
        Prepares features and fits the underlying estimator.
        Handles eval_set transformation for boosting models.
        """
        X_prepared = self._prepare_X(X)

        # Handle eval_set if present (common in XGB/LGBM)
        if "eval_set" in kwargs and kwargs["eval_set"] is not None:
            processed_eval_set = []
            for X_val, y_val in kwargs["eval_set"]:
                processed_eval_set.append((self._prepare_X(X_val), y_val))
            kwargs["eval_set"] = processed_eval_set

        self.estimator.fit(X_prepared, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        X_prepared = self._prepare_X(X)
        return self.estimator.predict(X_prepared, **kwargs)

    def predict_proba(self, X, **kwargs):
        X_prepared = self._prepare_X(X)
        return self.estimator.predict_proba(X_prepared, **kwargs)

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_


def get_learner(name):
    """
    Factory function to instantiate models based on the Hept-View architecture.

    Args:
        name (str): The name of the learner to instantiate.

    Returns:
        FeatureBinder or BaseEstimator: The configured model.
    """

    # 1. Sparse Lexical Branch (Text Modality)
    if name == "LexicalBagger":
        model = RandomForestClassifier(**LEXICAL_BAGGER_PARAMS)
        # Input: Sparse Lexical + Dense Metadata
        return FeatureBinder(model, feature_keys=["X_lexical", "X_meta"])

    # 2. Sparse Behavioral Branch (History Modality)
    elif name == "CommunityBagger":
        model = RandomForestClassifier(**COMMUNITY_BAGGER_PARAMS)
        # Input: Sparse Behavioral + Dense Metadata
        return FeatureBinder(model, feature_keys=["X_behavioral", "X_meta"])

    # 3. Dense Semantic Branch (Text Modality) - Volatile (XGB)
    elif name == "SemanticBooster":
        model = XGBClassifier(**SEMANTIC_BOOSTER_PARAMS)
        # Input: Dense Semantic + Dense Metadata
        return FeatureBinder(model, feature_keys=["X_semantic", "X_meta"])

    # 4. Dense Semantic Branch (Text Modality) - Volatile (LGBM)
    elif name == "SemanticGradient":
        model = LGBMClassifier(**SEMANTIC_GRADIENT_PARAMS)
        # Input: Dense Semantic + Dense Metadata
        return FeatureBinder(model, feature_keys=["X_semantic", "X_meta"])

    # 5. Dense Semantic Branch (Text Modality) - Stable (RF)
    elif name == "SemanticBagger":
        model = RandomForestClassifier(**SEMANTIC_BAGGER_PARAMS)
        # Input: Dense Semantic + Dense Metadata
        return FeatureBinder(model, feature_keys=["X_semantic", "X_meta"])

    # 6. Contextual Branch (Metadata Modality) - Stable (Linear)
    elif name == "MetadataAnchor":
        model = LogisticRegression(**METADATA_ANCHOR_PARAMS)
        # Input: Dense Metadata only
        return FeatureBinder(model, feature_keys=["X_meta"])

    # 7. Contextual Branch (Metadata Modality) - Volatile (Non-linear)
    elif name == "TemporalBooster":
        model = LGBMClassifier(**TEMPORAL_BOOSTER_PARAMS)
        # Input: Dense Metadata only
        return FeatureBinder(model, feature_keys=["X_meta"])

    # Level 2: Meta-Learner
    elif name == "MetaLearner":
        # The meta-learner receives a simple numpy array of predictions,
        # so no FeatureBinder is needed (or keys are implicit).
        return LogisticRegression(**META_LEARNER_PARAMS)

    else:
        raise ValueError(f"Unknown learner name: {name}")
