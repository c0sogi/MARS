import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import setup_logger, set_seed

logger = setup_logger("model_pipeline")


class BaseViewLearner:
    """
    A wrapper for a base learner (Random Forest) operating on a specific data view.
    Handles feature concatenation (View + Meta) and model training.
    """

    def __init__(self, view_key, is_sparse_view=True):
        """
        Args:
            view_key (str): The key in the feature dictionary ('lexical', 'semantic', 'behavioral').
            is_sparse_view (bool): Whether the view features are sparse (requires sparse.hstack).
        """
        self.view_key = view_key
        self.is_sparse_view = is_sparse_view
        self.model = RandomForestClassifier(**Config.RF_PARAMS)

    def _prepare_features(self, X_dict):
        """
        Concatenates the specific view features with the metadata features.
        """
        view_feats = X_dict[self.view_key]
        meta_feats = X_dict["meta"]

        # Ensure meta features are 2D
        if len(meta_feats.shape) == 1:
            meta_feats = meta_feats.reshape(-1, 1)

        if self.is_sparse_view:
            # Convert meta to sparse if needed or just hstack using scipy
            # scipy.sparse.hstack handles mixing sparse and dense arrays efficiently
            combined = sparse.hstack([view_feats, meta_feats], format="csr")
        else:
            # Both are dense
            combined = np.hstack([view_feats, meta_feats])

        return combined

    def fit(self, X_dict, y):
        X_combined = self._prepare_features(X_dict)
        self.model.fit(X_combined, y)
        return self

    def predict_proba(self, X_dict):
        X_combined = self._prepare_features(X_dict)
        # Return probability of positive class (1)
        return self.model.predict_proba(X_combined)[:, 1]


class TriViewStackingClassifier:
    """
    Implements the Tri-View Stacking Ensemble.
    Level 1: Lexical RF, Semantic RF, Behavioral RF.
    Level 2: Logistic Regression Meta-Learner.
    """

    def __init__(self):
        set_seed(Config.SEED)

        # Initialize Base Learners
        self.lexical_learner = BaseViewLearner(view_key="lexical", is_sparse_view=True)
        self.semantic_learner = BaseViewLearner(
            view_key="semantic", is_sparse_view=False
        )
        self.behavioral_learner = BaseViewLearner(
            view_key="behavioral", is_sparse_view=True
        )

        # Initialize Meta Learner
        self.meta_learner = LogisticRegression(**Config.META_LEARNER_PARAMS)

        # Scaler for meta-learner inputs (probabilities)
        # Logistic Regression benefits from scaling, though probs are bounded [0,1].
        # We'll use it to be safe and consistent.
        self.meta_scaler = StandardScaler()

        self.is_fitted = False

    def _get_fold_data(self, X_dict, train_idx, val_idx):
        """Helper to slice the dictionary of features."""
        X_train_fold = {}
        X_val_fold = {}

        for key, data in X_dict.items():
            # Support both sparse matrices and numpy arrays
            if sparse.issparse(data):
                X_train_fold[key] = data[train_idx]
                X_val_fold[key] = data[val_idx]
            else:
                X_train_fold[key] = data[train_idx]
                X_val_fold[key] = data[val_idx]

        return X_train_fold, X_val_fold

    def fit_cv(self, X_train_dict, y_train):
        """
        Performs K-Fold Cross-Validation to:
        1. Generate OOF predictions for the Meta-Learner.
        2. Train the Meta-Learner.
        3. Retrain Base Learners on the full dataset.
        """
        logger.info(f"Starting {Config.N_FOLDS}-Fold Cross-Validation Stacking...")

        # Ensure y is numpy array
        y_train = np.array(y_train)
        n_samples = len(y_train)

        # Storage for OOF predictions: [n_samples, 3] (Lexical, Semantic, Behavioral)
        oof_preds = np.zeros((n_samples, 3))

        kf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # ---------------------------------------------------------
        # Phase 1: OOF Generation
        # ---------------------------------------------------------
        for fold, (train_idx, val_idx) in enumerate(
            kf.split(np.zeros(n_samples), y_train)
        ):
            logger.info(f"Processing Fold {fold + 1}/{Config.N_FOLDS}")

            # Slice data
            X_tr_fold, X_val_fold = self._get_fold_data(
                X_train_dict, train_idx, val_idx
            )
            y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Temporary learners for this fold
            lex_fold = BaseViewLearner("lexical", True).fit(X_tr_fold, y_tr_fold)
            sem_fold = BaseViewLearner("semantic", False).fit(X_tr_fold, y_tr_fold)
            beh_fold = BaseViewLearner("behavioral", True).fit(X_tr_fold, y_tr_fold)

            # Predict on validation fold
            p_lex = lex_fold.predict_proba(X_val_fold)
            p_sem = sem_fold.predict_proba(X_val_fold)
            p_beh = beh_fold.predict_proba(X_val_fold)

            # Store OOF
            oof_preds[val_idx, 0] = p_lex
            oof_preds[val_idx, 1] = p_sem
            oof_preds[val_idx, 2] = p_beh

            # Fold Metrics
            fold_auc_lex = roc_auc_score(y_val_fold, p_lex)
            fold_auc_sem = roc_auc_score(y_val_fold, p_sem)
            fold_auc_beh = roc_auc_score(y_val_fold, p_beh)

            logger.info(
                f"Fold {fold+1} AUCs - Lexical: {fold_auc_lex}, Semantic: {fold_auc_sem}, Behavioral: {fold_auc_beh}"
            )

        # ---------------------------------------------------------
        # Phase 2: Train Meta-Learner
        # ---------------------------------------------------------
        logger.info("Training Meta-Learner on OOF predictions...")

        # Calculate overall OOF AUC for base models
        oof_auc_lex = roc_auc_score(y_train, oof_preds[:, 0])
        oof_auc_sem = roc_auc_score(y_train, oof_preds[:, 1])
        oof_auc_beh = roc_auc_score(y_train, oof_preds[:, 2])

        logger.info(f"Overall OOF AUC - Lexical: {oof_auc_lex}")
        logger.info(f"Overall OOF AUC - Semantic: {oof_auc_sem}")
        logger.info(f"Overall OOF AUC - Behavioral: {oof_auc_beh}")

        # Scale OOF predictions
        X_meta_train = self.meta_scaler.fit_transform(oof_preds)

        # Fit Meta-Learner
        self.meta_learner.fit(X_meta_train, y_train)

        # Check Meta-Learner performance on OOF (sanity check, slightly biased as it saw these labels)
        meta_oof_preds = self.meta_learner.predict_proba(X_meta_train)[:, 1]
        meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
        logger.info(f"Meta-Learner OOF AUC (Stacked): {meta_oof_auc}")

        # Print coefficients to see contribution of each view
        coeffs = self.meta_learner.coef_[0]
        logger.info(
            f"Meta-Learner Coefficients: Lexical={coeffs[0]:.4f}, Semantic={coeffs[1]:.4f}, Behavioral={coeffs[2]:.4f}"
        )

        # ---------------------------------------------------------
        # Phase 3: Retrain Base Learners on Full Data
        # ---------------------------------------------------------
        logger.info("Retraining Base Learners on full training set...")
        self.lexical_learner.fit(X_train_dict, y_train)
        self.semantic_learner.fit(X_train_dict, y_train)
        self.behavioral_learner.fit(X_train_dict, y_train)

        self.is_fitted = True
        logger.info("Training complete.")

    def predict_proba(self, X_dict):
        """
        Generates final predictions for test data.
        1. Predict with retrained base learners.
        2. Stack predictions.
        3. Predict with meta-learner.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted using fit_cv before predicting.")

        # 1. Base Learner Predictions
        p_lex = self.lexical_learner.predict_proba(X_dict)
        p_sem = self.semantic_learner.predict_proba(X_dict)
        p_beh = self.behavioral_learner.predict_proba(X_dict)

        # 2. Stack
        X_stack = np.column_stack([p_lex, p_sem, p_beh])

        # 3. Scale
        X_stack_scaled = self.meta_scaler.transform(X_stack)

        # 4. Meta Prediction
        final_probs = self.meta_learner.predict_proba(X_stack_scaled)[:, 1]

        return final_probs
