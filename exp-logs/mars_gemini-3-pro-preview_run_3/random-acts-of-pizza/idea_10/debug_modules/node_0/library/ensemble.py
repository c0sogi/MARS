import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

from library.config import (
    RF_LEXICAL_PARAMS,
    RF_BEHAVIORAL_PARAMS,
    XGB_CONTEXTUAL_PARAMS,
    XGB_FIT_PARAMS,
    STACKING_META_PARAMS,
    NUM_FOLDS,
    SEED,
    WORKING_DIR,
)
from library.utils import Timer, ensure_dir


class TriViewStackingEnsemble:
    def __init__(self):
        """
        Initializes the Tri-View Stacking Ensemble with Level 1 and Level 2 models
        configured via the global configuration file.
        """
        # Level 1 Models
        # View 1: Lexical (Sparse Text) -> Random Forest
        self.rf_lexical = RandomForestClassifier(**RF_LEXICAL_PARAMS)

        # View 2: Behavioral (Sparse Subreddits) -> Random Forest
        self.rf_behavioral = RandomForestClassifier(**RF_BEHAVIORAL_PARAMS)

        # View 3: Contextual (Dense Embeddings + Meta) -> XGBoost
        self.xgb_contextual = XGBClassifier(**XGB_CONTEXTUAL_PARAMS)

        # Level 2 Meta-Learner
        self.meta_learner = LogisticRegression(**STACKING_META_PARAMS)

        # State flags
        self.is_base_fitted = False
        self.is_meta_fitted = False

    def _slice_data(self, X_dict, indices):
        """
        Helper method to slice the dictionary of feature matrices/arrays.
        """
        return {
            "lexical": X_dict["lexical"][indices],
            "behavioral": X_dict["behavioral"][indices],
            "dense": X_dict["dense"][indices],
        }

    def get_oof_predictions(self, X_train_dict, y_train):
        """
        Performs Stratified K-Fold Cross Validation to generate Out-Of-Fold (OOF) predictions.
        These OOF predictions serve as the training data for the Level 2 Meta-Learner.

        Args:
            X_train_dict (dict): Dictionary containing 'lexical', 'behavioral', and 'dense' features.
            y_train (array-like): Target labels.

        Returns:
            np.ndarray: OOF prediction matrix of shape (n_samples, 3).
        """
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

        n_samples = len(y_train)
        # Columns: 0=Lexical, 1=Behavioral, 2=Contextual
        oof_preds = np.zeros((n_samples, 3))

        print(f"Starting {NUM_FOLDS}-Fold Cross-Validation for Stacking...")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            with Timer(f"Fold {fold + 1}"):
                # Prepare Fold Data
                X_tr = self._slice_data(X_train_dict, train_idx)
                y_tr = y_train[train_idx]

                X_val = self._slice_data(X_train_dict, val_idx)
                y_val = y_train[val_idx]

                # --- 1. Train Lexical Bagger (RF) ---
                self.rf_lexical.fit(X_tr["lexical"], y_tr)
                p_lex = self.rf_lexical.predict_proba(X_val["lexical"])[:, 1]

                # --- 2. Train Behavioral Bagger (RF) ---
                self.rf_behavioral.fit(X_tr["behavioral"], y_tr)
                p_beh = self.rf_behavioral.predict_proba(X_val["behavioral"])[:, 1]

                # --- 3. Train Contextual Booster (XGB) ---
                # XGBoost requires a validation set for early stopping
                self.xgb_contextual.fit(
                    X_tr["dense"],
                    y_tr,
                    eval_set=[(X_val["dense"], y_val)],
                    **XGB_FIT_PARAMS,
                )
                p_ctx = self.xgb_contextual.predict_proba(X_val["dense"])[:, 1]

                # Store Predictions
                oof_preds[val_idx, 0] = p_lex
                oof_preds[val_idx, 1] = p_beh
                oof_preds[val_idx, 2] = p_ctx

                # Logging
                auc_lex = roc_auc_score(y_val, p_lex)
                auc_beh = roc_auc_score(y_val, p_beh)
                auc_ctx = roc_auc_score(y_val, p_ctx)
                print(
                    f"  Fold {fold+1} AUCs - Lexical: {auc_lex}, Behavioral: {auc_beh}, Contextual: {auc_ctx}"
                )

        # Calculate Global OOF AUCs
        total_auc_lex = roc_auc_score(y_train, oof_preds[:, 0])
        total_auc_beh = roc_auc_score(y_train, oof_preds[:, 1])
        total_auc_ctx = roc_auc_score(y_train, oof_preds[:, 2])

        print(
            f"Global OOF AUCs - Lexical: {total_auc_lex}, Behavioral: {total_auc_beh}, Contextual: {total_auc_ctx}"
        )

        return oof_preds

    def fit_meta_learner(self, oof_preds, y_train):
        """
        Fits the Level 2 Logistic Regression on the OOF predictions.

        Args:
            oof_preds (np.ndarray): The (n_samples, 3) matrix from get_oof_predictions.
            y_train (array-like): Target labels.
        """
        with Timer("Fit Meta-Learner"):
            self.meta_learner.fit(oof_preds, y_train)
            self.is_meta_fitted = True

            # Evaluate In-Sample Fit
            meta_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
            score = roc_auc_score(y_train, meta_preds)
            print(f"Meta-Learner CV Score (on OOF): {score}")
            print(f"Meta-Learner Coefficients: {self.meta_learner.coef_}")

    def refit_base_models(self, X_train_dict, y_train):
        """
        Refits all Level 1 models on the full training dataset.
        For XGBoost, creates a small internal validation split to enable early stopping.
        """
        with Timer("Refit Base Models"):
            # 1. Lexical RF
            self.rf_lexical.fit(X_train_dict["lexical"], y_train)

            # 2. Behavioral RF
            self.rf_behavioral.fit(X_train_dict["behavioral"], y_train)

            # 3. Contextual XGB
            # We need a validation set for early stopping even when refitting on "full" train.
            # We use a 10% stratified split. This is a trade-off: we lose 10% of data for training
            # but ensure the model doesn't overfit (which XGBoost is prone to without early stopping).
            X_tr_dense, X_val_dense, y_tr, y_val = train_test_split(
                X_train_dict["dense"],
                y_train,
                test_size=0.1,
                random_state=SEED,
                stratify=y_train,
            )

            self.xgb_contextual.fit(
                X_tr_dense, y_tr, eval_set=[(X_val_dense, y_val)], **XGB_FIT_PARAMS
            )

            self.is_base_fitted = True

    def predict(self, X_test_dict):
        """
        Generates final predictions for the test set by passing data through
        Level 1 models and then the Level 2 meta-learner.

        Args:
            X_test_dict (dict): Dictionary of test features.

        Returns:
            np.ndarray: Probability of success.
        """
        if not self.is_base_fitted or not self.is_meta_fitted:
            raise RuntimeError(
                "Models must be fitted before prediction. Run get_oof_predictions, fit_meta_learner, and refit_base_models."
            )

        with Timer("Ensemble Prediction"):
            # Level 1 Inference
            p_lex = self.rf_lexical.predict_proba(X_test_dict["lexical"])[:, 1]
            p_beh = self.rf_behavioral.predict_proba(X_test_dict["behavioral"])[:, 1]
            p_ctx = self.xgb_contextual.predict_proba(X_test_dict["dense"])[:, 1]

            # Stack Predictions
            X_meta = np.column_stack((p_lex, p_beh, p_ctx))

            # Level 2 Inference
            final_probs = self.meta_learner.predict_proba(X_meta)[:, 1]

        return final_probs

    def save_model(self, path):
        """Saves the ensemble object to disk."""
        ensure_dir(path)
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path):
        """Loads the ensemble object from disk."""
        return joblib.load(path)
