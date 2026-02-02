import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config
from library.utils import setup_logger


class LexicalBagger:
    """
    Level 1 Base Learner: Sparse Lexical Branch.
    Uses Random Forest on TF-IDF Text features concatenated with Global Metadata.
    Designed to capture specific keywords while regularizing with metadata anchors.
    """

    def __init__(self):
        self.logger = setup_logger("LexicalBagger")
        self.model = RandomForestClassifier(**Config.SPARSE_RF_PARAMS)

    def fit(self, X_lexical, X_metadata, y):
        """
        Args:
            X_lexical: Sparse TF-IDF matrix of request text.
            X_metadata: Dense numpy array of metadata features.
            y: Target labels.
        """
        # Convert metadata to sparse for efficient stacking with high-dim text
        X_meta_sparse = sp.csr_matrix(X_metadata)
        X_combined = sp.hstack([X_lexical, X_meta_sparse])

        self.logger.info(f"Training LexicalBagger on shape {X_combined.shape}")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_lexical, X_metadata):
        X_meta_sparse = sp.csr_matrix(X_metadata)
        X_combined = sp.hstack([X_lexical, X_meta_sparse])
        return self.model.predict_proba(X_combined)[:, 1]


class CommunityBagger:
    """
    Level 1 Base Learner: Sparse Behavioral Branch.
    Uses Random Forest on TF-IDF Subreddit history concatenated with Global Metadata.
    Treats user history as a 'bag-of-concepts' to capture niche community signals.
    """

    def __init__(self):
        self.logger = setup_logger("CommunityBagger")
        self.model = RandomForestClassifier(**Config.SPARSE_RF_PARAMS)

    def fit(self, X_behavioral, X_metadata, y):
        """
        Args:
            X_behavioral: Sparse TF-IDF matrix of subreddit history.
            X_metadata: Dense numpy array of metadata features.
            y: Target labels.
        """
        X_meta_sparse = sp.csr_matrix(X_metadata)
        X_combined = sp.hstack([X_behavioral, X_meta_sparse])

        self.logger.info(f"Training CommunityBagger on shape {X_combined.shape}")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_behavioral, X_metadata):
        X_meta_sparse = sp.csr_matrix(X_metadata)
        X_combined = sp.hstack([X_behavioral, X_meta_sparse])
        return self.model.predict_proba(X_combined)[:, 1]


class SemanticBooster:
    """
    Level 1 Base Learner: Dense Semantic Branch (Boosting).
    Uses XGBoost on Dense Embeddings concatenated with Global Metadata.

    Implements 'Validation-Guided Retraining':
    - Can accept an external validation set (eval_set) to trigger early stopping.
    - This is crucial for preventing overfitting on the dense embedding space.
    """

    def __init__(self):
        self.logger = setup_logger("SemanticBooster")
        self.model = XGBClassifier(**Config.XGB_PARAMS)

    def fit(self, X_semantic, X_metadata, y, eval_set=None):
        """
        Args:
            X_semantic: Dense numpy array of text embeddings.
            X_metadata: Dense numpy array of metadata features.
            y: Target labels.
            eval_set: Optional tuple (X_sem_val, X_meta_val, y_val) for early stopping.
        """
        X_combined = np.hstack([X_semantic, X_metadata])

        fit_params = {"verbose": False}

        if eval_set:
            # Unpack and prepare validation set
            X_sem_val, X_meta_val, y_val = eval_set
            X_val_combined = np.hstack([X_sem_val, X_meta_val])

            fit_params["eval_set"] = [(X_val_combined, y_val)]
            self.model.set_params(
                early_stopping_rounds=Config.XGB_EARLY_STOPPING_ROUNDS
            )
            self.logger.info(
                f"Training SemanticBooster with early stopping on shape {X_combined.shape}"
            )
        else:
            self.model.set_params(early_stopping_rounds=None)
            self.logger.info(f"Training SemanticBooster on shape {X_combined.shape}")

        self.model.fit(X_combined, y, **fit_params)
        return self

    def predict_proba(self, X_semantic, X_metadata):
        X_combined = np.hstack([X_semantic, X_metadata])
        return self.model.predict_proba(X_combined)[:, 1]


class SemanticBagger:
    """
    Level 1 Base Learner: Dense Semantic Branch (Bagging).
    Uses Random Forest on Dense Embeddings concatenated with Global Metadata.

    Applies STRICT depth constraints (via DENSE_RF_PARAMS) to prevents the tree
    from memorizing noise in the continuous embedding space.
    """

    def __init__(self):
        self.logger = setup_logger("SemanticBagger")
        self.model = RandomForestClassifier(**Config.DENSE_RF_PARAMS)

    def fit(self, X_semantic, X_metadata, y):
        X_combined = np.hstack([X_semantic, X_metadata])
        self.logger.info(f"Training SemanticBagger on shape {X_combined.shape}")
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_semantic, X_metadata):
        X_combined = np.hstack([X_semantic, X_metadata])
        return self.model.predict_proba(X_combined)[:, 1]


class MetadataAnchor:
    """
    Level 1 Base Learner: Contextual Branch.
    Uses Logistic Regression on Global Metadata only.
    Acts as a high-bias regularizer to stabilize the ensemble.
    """

    def __init__(self):
        self.logger = setup_logger("MetadataAnchor")
        self.model = LogisticRegression(**Config.LINEAR_PARAMS)

    def fit(self, X_metadata, y):
        self.logger.info(f"Training MetadataAnchor on shape {X_metadata.shape}")
        self.model.fit(X_metadata, y)
        return self

    def predict_proba(self, X_metadata):
        return self.model.predict_proba(X_metadata)[:, 1]


class StackingMetaLearner:
    """
    Level 2 Meta-Learner.
    Uses Logistic Regression to calibrate and combine predictions from the five Level 1 learners.
    """

    def __init__(self):
        self.logger = setup_logger("StackingMetaLearner")
        self.model = LogisticRegression(**Config.LINEAR_PARAMS)

    def fit(self, X_preds, y):
        """
        Args:
            X_preds: Matrix of shape (n_samples, 5) containing probas from base learners.
            y: Target labels.
        """
        self.logger.info(f"Training StackingMetaLearner on shape {X_preds.shape}")
        self.model.fit(X_preds, y)
        return self

    def predict_proba(self, X_preds):
        return self.model.predict_proba(X_preds)[:, 1]
