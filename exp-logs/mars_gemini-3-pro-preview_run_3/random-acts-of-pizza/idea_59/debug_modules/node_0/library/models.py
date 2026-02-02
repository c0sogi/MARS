import os
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

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
from library.utils import get_logger

logger = get_logger("models")


class BaseModel:
    """
    Abstract base class for Level 1 models.
    Handles initialization, saving, loading, and common fit/predict logic.
    """

    def __init__(self, name, params, model_class):
        self.name = name
        self.params = params.copy()
        self.model_class = model_class
        self.model = self._init_model()

    def _init_model(self):
        return self.model_class(**self.params)

    def _prepare_data(self, features_dict):
        """
        Selects and combines features from the dictionary.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def fit(self, X_train_dict, y_train, X_val_dict=None, y_val=None):
        """
        Fits the model. Handles validation sets for early stopping if supported.
        """
        X_train = self._prepare_data(X_train_dict)

        fit_params = {}

        # Handle Validation Data for Volatile Learners (XGB/LGBM)
        if X_val_dict is not None and y_val is not None:
            is_xgb = "XGBClassifier" in self.model_class.__name__
            is_lgbm = "LGBMClassifier" in self.model_class.__name__

            if is_xgb or is_lgbm:
                X_val = self._prepare_data(X_val_dict)
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["eval_metric"] = "auc"

                # LightGBM specific early stopping via callbacks
                if is_lgbm:
                    # Use a conservative default if not in params, though config usually rules
                    rounds = self.params.get("early_stopping_rounds", 50)
                    fit_params["callbacks"] = [
                        early_stopping(stopping_rounds=rounds, verbose=False)
                    ]

                # XGBoost handles early_stopping_rounds in constructor or fit
                # It is already in SEMANTIC_BOOSTER_PARAMS for XGB

        logger.info(
            f"Training {self.name} on {X_train.shape[0]} samples (Features: {X_train.shape[1]})..."
        )
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict_proba(self, features_dict):
        X = self._prepare_data(features_dict)
        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]

    def save(self, directory):
        path = os.path.join(directory, f"{self.name}.joblib")
        joblib.dump(self.model, path)

    def load(self, directory):
        path = os.path.join(directory, f"{self.name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            logger.warning(f"Model file not found at {path}")


# =============================================================================
# Level 1: Base Learners
# =============================================================================


class LexicalBagger(BaseModel):
    def __init__(self):
        super().__init__(
            "lexical_bagger", LEXICAL_BAGGER_PARAMS, RandomForestClassifier
        )

    def _prepare_data(self, features_dict):
        # Sparse Lexical + Dense Metadata -> Sparse
        return sp.hstack([features_dict["lexical"], features_dict["metadata"]])


class CommunityBagger(BaseModel):
    def __init__(self):
        super().__init__(
            "community_bagger", COMMUNITY_BAGGER_PARAMS, RandomForestClassifier
        )

    def _prepare_data(self, features_dict):
        # Sparse Behavioral + Dense Metadata -> Sparse
        return sp.hstack([features_dict["behavioral"], features_dict["metadata"]])


class SemanticBooster(BaseModel):
    def __init__(self):
        super().__init__("semantic_booster", SEMANTIC_BOOSTER_PARAMS, XGBClassifier)

    def _prepare_data(self, features_dict):
        # Dense Semantic + Dense Metadata -> Dense
        return np.hstack([features_dict["semantic"], features_dict["metadata"]])


class SemanticGradient(BaseModel):
    def __init__(self):
        super().__init__("semantic_gradient", SEMANTIC_GRADIENT_PARAMS, LGBMClassifier)

    def _prepare_data(self, features_dict):
        # Dense Semantic + Dense Metadata -> Dense
        return np.hstack([features_dict["semantic"], features_dict["metadata"]])


class SemanticBagger(BaseModel):
    def __init__(self):
        super().__init__(
            "semantic_bagger", SEMANTIC_BAGGER_PARAMS, RandomForestClassifier
        )

    def _prepare_data(self, features_dict):
        # Dense Semantic + Dense Metadata -> Dense
        return np.hstack([features_dict["semantic"], features_dict["metadata"]])


class MetadataAnchor(BaseModel):
    def __init__(self):
        super().__init__("metadata_anchor", METADATA_ANCHOR_PARAMS, LogisticRegression)

    def _prepare_data(self, features_dict):
        # Metadata only
        return features_dict["metadata"]


class TemporalBooster(BaseModel):
    def __init__(self):
        super().__init__("temporal_booster", TEMPORAL_BOOSTER_PARAMS, LGBMClassifier)

    def _prepare_data(self, features_dict):
        # Metadata only
        return features_dict["metadata"]


# =============================================================================
# Level 2: Meta Learner
# =============================================================================


class MetaLearner:
    """
    Level 2 Stacking Meta-Learner.
    Trained on the OOF predictions of the Level 1 models.
    """

    def __init__(self):
        self.name = "meta_learner"
        self.params = META_LEARNER_PARAMS.copy()
        self.model = LogisticRegression(**self.params)

    def fit(self, X_preds, y):
        """
        Args:
            X_preds: Matrix of shape (n_samples, n_models) containing Level 1 predictions.
            y: Target labels.
        """
        logger.info(f"Training Meta-Learner on shape {X_preds.shape}...")
        self.model.fit(X_preds, y)
        return self

    def predict_proba(self, X_preds):
        return self.model.predict_proba(X_preds)[:, 1]

    def save(self, directory):
        path = os.path.join(directory, f"{self.name}.joblib")
        joblib.dump(self.model, path)

    def load(self, directory):
        path = os.path.join(directory, f"{self.name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
