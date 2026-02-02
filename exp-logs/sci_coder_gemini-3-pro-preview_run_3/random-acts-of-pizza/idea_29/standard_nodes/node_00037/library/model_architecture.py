import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config
from library.utils import setup_logger


class PentViewEnsemble:
    """
    Implements the Pent-View Stacking Ensemble architecture.
    Consists of 5 Level-1 Base Learners and 1 Level-2 Meta-Learner.
    Uses a CV-Bagging Inference strategy where the Meta-Learner is calibrated
    on the validation fold of each split.
    """

    def __init__(self):
        self.logger = setup_logger("PentViewEnsemble")
        self.fitted = False

        # =========================================================================
        # Level 1: Base Learners
        # =========================================================================

        # 1. Sparse Lexical Branch (Text Modality)
        # Random Forest on TF-IDF + Metadata
        self.lexical_bagger = RandomForestClassifier(**Config.RF_PARAMS)

        # 2. Sparse Behavioral Branch (History Modality)
        # Random Forest on Subreddit TF-IDF + Metadata
        self.community_bagger = RandomForestClassifier(**Config.RF_PARAMS)

        # 3. Dense Semantic Branch (Text Modality - Boosting)
        # XGBoost on Embeddings + Metadata
        # Note: early_stopping_rounds is passed via **Config.XGB_PARAMS to __init__
        self.semantic_booster = XGBClassifier(**Config.XGB_PARAMS)

        # 4. Dense Semantic Branch (Text Modality - Bagging)
        # Random Forest on Embeddings + Metadata
        self.semantic_bagger = RandomForestClassifier(**Config.RF_PARAMS)

        # 5. Contextual Branch (Metadata Modality)
        # Logistic Regression on Metadata only
        self.metadata_anchor = LogisticRegression(**Config.LR_ANCHOR_PARAMS)

        # =========================================================================
        # Level 2: Meta-Learner
        # =========================================================================
        # Logistic Regression to calibrate ensemble weights
        self.meta_learner = LogisticRegression(**Config.META_LEARNER_PARAMS)

    def fit(self, train_views, y_train, val_views, y_val):
        """
        Trains the ensemble on the training fold and calibrates the meta-learner
        using the validation fold.

        Args:
            train_views (dict): Dictionary containing 'lexical', 'behavioral',
                                'semantic', 'metadata' for training.
            y_train (array-like): Training labels.
            val_views (dict): Dictionary containing views for validation.
                              Used for XGBoost Early Stopping and Meta-Learner training.
            y_val (array-like): Validation labels.
        """
        self.logger.info("Starting training of Level 1 Base Learners...")

        # --- Train Level 1 Models ---

        # 1. Lexical Bagger
        self.logger.info("Training Lexical Bagger (RF)...")
        self.lexical_bagger.fit(train_views["lexical"], y_train)

        # 2. Community Bagger
        self.logger.info("Training Community Bagger (RF)...")
        self.community_bagger.fit(train_views["behavioral"], y_train)

        # 3. Semantic Booster
        self.logger.info("Training Semantic Booster (XGB)...")
        # XGBoost requires eval_set for early stopping
        # We suppress verbose output to keep logs clean
        self.semantic_booster.fit(
            train_views["semantic"],
            y_train,
            eval_set=[(val_views["semantic"], y_val)],
            verbose=False,
        )

        # 4. Semantic Bagger
        self.logger.info("Training Semantic Bagger (RF)...")
        self.semantic_bagger.fit(train_views["semantic"], y_train)

        # 5. Metadata Anchor
        self.logger.info("Training Metadata Anchor (LR)...")
        self.metadata_anchor.fit(train_views["metadata"], y_train)

        # --- Train Level 2 Meta-Learner ---
        self.logger.info("Generating Validation Predictions for Meta-Learner...")

        # Generate OOF/Validation predictions (probability of class 1)
        # These act as the input features for the meta-learner
        p_lex = self.lexical_bagger.predict_proba(val_views["lexical"])[:, 1]
        p_com = self.community_bagger.predict_proba(val_views["behavioral"])[:, 1]
        p_sem_boost = self.semantic_booster.predict_proba(val_views["semantic"])[:, 1]
        p_sem_bag = self.semantic_bagger.predict_proba(val_views["semantic"])[:, 1]
        p_meta = self.metadata_anchor.predict_proba(val_views["metadata"])[:, 1]

        # Stack predictions column-wise: Shape (N_val, 5)
        X_meta = np.column_stack([p_lex, p_com, p_sem_boost, p_sem_bag, p_meta])

        self.logger.info(
            f"Training Meta-Learner on stacked features shape: {X_meta.shape}"
        )
        self.meta_learner.fit(X_meta, y_val)

        self.fitted = True
        self.logger.info("Ensemble training complete.")

    def predict_proba(self, views):
        """
        Generates probability predictions for the positive class (Received Pizza).

        Args:
            views (dict): Dictionary containing feature views ('lexical', 'behavioral',
                          'semantic', 'metadata').

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble must be fitted before calling predict_proba.")

        # 1. Generate Level 1 Predictions
        p_lex = self.lexical_bagger.predict_proba(views["lexical"])[:, 1]
        p_com = self.community_bagger.predict_proba(views["behavioral"])[:, 1]
        p_sem_boost = self.semantic_booster.predict_proba(views["semantic"])[:, 1]
        p_sem_bag = self.semantic_bagger.predict_proba(views["semantic"])[:, 1]
        p_meta = self.metadata_anchor.predict_proba(views["metadata"])[:, 1]

        # 2. Stack Predictions
        X_meta = np.column_stack([p_lex, p_com, p_sem_boost, p_sem_bag, p_meta])

        # 3. Generate Level 2 Prediction
        return self.meta_learner.predict_proba(X_meta)[:, 1]
