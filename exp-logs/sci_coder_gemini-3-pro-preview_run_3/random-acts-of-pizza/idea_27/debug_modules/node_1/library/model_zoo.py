import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from library.config import Config


class BaseLearner:
    """
    Abstract base class for all Level 1 learners in the stacking ensemble.
    Enforces a consistent interface for fitting and probability prediction.
    """

    def fit(self, X_main, X_meta, y, eval_set=None):
        """
        Trains the model.

        Args:
            X_main: The primary feature view (Sparse matrix or Dense array).
                    Can be None for models that only use metadata.
            X_meta: The dense metadata feature matrix.
            y: Target labels.
            eval_set: Optional tuple (X_val_main, X_val_meta, y_val) for early stopping.
        """
        raise NotImplementedError("Fit method must be implemented by subclasses.")

    def predict_proba(self, X_main, X_meta):
        """
        Predicts class probabilities.

        Args:
            X_main: The primary feature view.
            X_meta: The dense metadata feature matrix.

        Returns:
            np.ndarray: Probability of the positive class (class 1).
        """
        raise NotImplementedError(
            "Predict_proba method must be implemented by subclasses."
        )


class LexicalBagger(BaseLearner):
    """
    Sparse Lexical Branch.
    Uses a Random Forest on the concatenation of Sparse TF-IDF (Text) and Dense Metadata.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_PARAMS)

    def fit(self, X_main, X_meta, y, eval_set=None):
        # Concatenate sparse text features with dense metadata
        # X_main is expected to be sparse (TF-IDF)
        X_combined = sparse.hstack([X_main, X_meta])
        self.model.fit(X_combined, y)

    def predict_proba(self, X_main, X_meta):
        X_combined = sparse.hstack([X_main, X_meta])
        return self.model.predict_proba(X_combined)[:, 1]


class BehavioralBagger(BaseLearner):
    """
    Sparse Behavioral Branch.
    Uses a Random Forest on the concatenation of Sparse TF-IDF (Subreddit History) and Dense Metadata.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_PARAMS)

    def fit(self, X_main, X_meta, y, eval_set=None):
        # Concatenate sparse history features with dense metadata
        X_combined = sparse.hstack([X_main, X_meta])
        self.model.fit(X_combined, y)

    def predict_proba(self, X_main, X_meta):
        X_combined = sparse.hstack([X_main, X_meta])
        return self.model.predict_proba(X_combined)[:, 1]


class SemanticBooster(BaseLearner):
    """
    Dense Semantic Branch (Boosting).
    Uses XGBoost on the concatenation of Dense Embeddings and Dense Metadata.
    Implements Validation-Guided Early Stopping.
    """

    def __init__(self):
        # We copy params to manipulate early_stopping_rounds safely
        self.params = Config.XGB_PARAMS.copy()

        # Extract early_stopping_rounds to pass to fit() explicitly
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", None)

        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_main, X_meta, y, eval_set=None):
        # Concatenate dense embeddings with dense metadata
        X_combined = np.hstack([X_main, X_meta])

        fit_params = {"verbose": False}

        if eval_set is not None and self.early_stopping_rounds is not None:
            # Unpack validation set and prepare it similarly
            X_val_main, X_val_meta, y_val = eval_set
            X_val_combined = np.hstack([X_val_main, X_val_meta])

            fit_params["eval_set"] = [(X_val_combined, y_val)]
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        else:
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(X_combined, y, **fit_params)

    def predict_proba(self, X_main, X_meta):
        X_combined = np.hstack([X_main, X_meta])
        return self.model.predict_proba(X_combined)[:, 1]


class SemanticBagger(BaseLearner):
    """
    Dense Semantic Branch (Bagging).
    Uses a Random Forest on the concatenation of Dense Embeddings and Dense Metadata.
    Provides diversity alongside the SemanticBooster.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**Config.RF_PARAMS)

    def fit(self, X_main, X_meta, y, eval_set=None):
        # Concatenate dense embeddings with dense metadata
        X_combined = np.hstack([X_main, X_meta])
        self.model.fit(X_combined, y)

    def predict_proba(self, X_main, X_meta):
        X_combined = np.hstack([X_main, X_meta])
        return self.model.predict_proba(X_combined)[:, 1]


class MetadataAnchor(BaseLearner):
    """
    Contextual Branch.
    Uses Logistic Regression on Global Metadata Vector only.
    Acts as a high-bias regularizer.
    Ignores X_main input.
    """

    def __init__(self):
        self.model = LogisticRegression(**Config.LR_PARAMS)

    def fit(self, X_main, X_meta, y, eval_set=None):
        # Ignores X_main (the primary modality) and uses only metadata
        self.model.fit(X_meta, y)

    def predict_proba(self, X_main, X_meta):
        # Ignores X_main
        return self.model.predict_proba(X_meta)[:, 1]
