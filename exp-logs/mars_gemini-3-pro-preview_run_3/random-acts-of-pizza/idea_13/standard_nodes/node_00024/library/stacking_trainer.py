import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import copy

from library.config import Config
from library.model_factory import (
    get_lexical_model,
    get_behavioral_model,
    get_semantic_model,
    get_meta_learner,
)
from library.utils import print_metric


class StackingEnsemble:
    """
    Implements the Contextually-Unified Topology-Matched Stacking Ensemble.

    Architecture:
    - Level 1:
        1. Lexical-Contextual Bagger (Random Forest) -> Sparse Text + Dense Metadata
        2. Behavioral-Contextual Bagger (Random Forest) -> Sparse History + Dense Metadata
        3. Semantic-Contextual Booster (XGBoost) -> Dense Embeddings + Dense Metadata
    - Level 2:
        - Meta-Learner (Logistic Regression) -> Aggregated Probabilities
    """

    def __init__(self):
        # Placeholders for final retrained models
        self.lexical_model = None
        self.behavioral_model = None
        self.semantic_model = None
        self.meta_learner = None

        # Placeholders for OOF scores
        self.oof_scores = {}

    def _get_fold_data(self, X, train_idx, val_idx):
        """Helper to slice data handling both Sparse Matrices and Numpy Arrays."""
        if sparse.issparse(X):
            return X[train_idx], X[val_idx]
        else:
            return X[train_idx], X[val_idx]

    def fit(self, X_lex, X_beh, X_sem, y):
        """
        Fits the stacking ensemble.

        1. Performs K-Fold CV to generate OOF predictions for Level 2 training.
        2. Trains Level 2 Meta-Learner on OOF predictions.
        3. Retrains all Level 1 models on the full dataset.

        Args:
            X_lex (sparse.csr_matrix): Lexical features (Text TF-IDF + Metadata).
            X_beh (sparse.csr_matrix): Behavioral features (Subreddit TF-IDF + Metadata).
            X_sem (np.ndarray): Semantic features (Embeddings + Metadata).
            y (np.ndarray): Target labels.
        """
        print(f"Starting Stacking Ensemble Training with {Config.N_FOLDS} folds...")

        # Initialize OOF arrays
        n_samples = y.shape[0]
        oof_lex = np.zeros(n_samples)
        oof_beh = np.zeros(n_samples)
        oof_sem = np.zeros(n_samples)

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
            print(f"  Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            # --- Slice Data ---
            X_lex_tr, X_lex_val = self._get_fold_data(X_lex, train_idx, val_idx)
            X_beh_tr, X_beh_val = self._get_fold_data(X_beh, train_idx, val_idx)
            X_sem_tr, X_sem_val = self._get_fold_data(X_sem, train_idx, val_idx)
            y_tr, y_val = y[train_idx], y[val_idx]

            # --- 1. Lexical Model (RF) ---
            model_lex = get_lexical_model()
            model_lex.fit(X_lex_tr, y_tr)
            # Predict proba for class 1
            oof_lex[val_idx] = model_lex.predict_proba(X_lex_val)[:, 1]

            # --- 2. Behavioral Model (RF) ---
            model_beh = get_behavioral_model()
            model_beh.fit(X_beh_tr, y_tr)
            oof_beh[val_idx] = model_beh.predict_proba(X_beh_val)[:, 1]

            # --- 3. Semantic Model (XGBoost) ---
            model_sem = get_semantic_model()
            # XGBoost supports early stopping with eval_set
            model_sem.fit(X_sem_tr, y_tr, eval_set=[(X_sem_val, y_val)], verbose=False)
            oof_sem[val_idx] = model_sem.predict_proba(X_sem_val)[:, 1]

        # --- Evaluate OOF Performance ---
        score_lex = roc_auc_score(y, oof_lex)
        score_beh = roc_auc_score(y, oof_beh)
        score_sem = roc_auc_score(y, oof_sem)

        self.oof_scores = {
            "Lexical": score_lex,
            "Behavioral": score_beh,
            "Semantic": score_sem,
        }

        print("\n--- Level 1 OOF Performance (AUC) ---")
        print_metric("Lexical View (RF)", score_lex)
        print_metric("Behavioral View (RF)", score_beh)
        print_metric("Semantic View (XGB)", score_sem)

        # --- Train Level 2 Meta-Learner ---
        print("\nTraining Level 2 Meta-Learner...")
        # Stack OOF predictions: Shape (N, 3)
        X_level2 = np.column_stack([oof_lex, oof_beh, oof_sem])

        self.meta_learner = get_meta_learner()
        self.meta_learner.fit(X_level2, y)

        # Check Meta-Learner Coefficients
        print("Meta-Learner Coefficients:", self.meta_learner.coef_[0])

        # --- Final Retraining of Level 1 Models ---
        print("\nRetraining Level 1 Models on Full Dataset...")

        # 1. Lexical
        self.lexical_model = get_lexical_model()
        self.lexical_model.fit(X_lex, y)

        # 2. Behavioral
        self.behavioral_model = get_behavioral_model()
        self.behavioral_model.fit(X_beh, y)

        # 3. Semantic
        # For the final XGBoost model, we need a validation set for early stopping
        # to prevent overfitting, or we rely on the robustness of the params.
        # We will use a small stratified holdout (10%) from the full train set
        # purely for the early stopping criteria of the final model.
        self.semantic_model = get_semantic_model()

        X_sem_train_final, X_sem_val_final, y_train_final, y_val_final = (
            train_test_split(
                X_sem, y, test_size=0.1, stratify=y, random_state=Config.SEED
            )
        )

        self.semantic_model.fit(
            X_sem_train_final,
            y_train_final,
            eval_set=[(X_sem_val_final, y_val_final)],
            verbose=False,
        )

        print("Training Complete.")

    def predict(self, X_lex, X_beh, X_sem):
        """
        Generates predictions for new data.

        Args:
            X_lex (sparse.csr_matrix): Lexical features.
            X_beh (sparse.csr_matrix): Behavioral features.
            X_sem (np.ndarray): Semantic features.

        Returns:
            np.ndarray: Predicted probabilities of class 1.
        """
        if any(
            m is None
            for m in [
                self.lexical_model,
                self.behavioral_model,
                self.semantic_model,
                self.meta_learner,
            ]
        ):
            raise RuntimeError("Models not trained. Call fit() first.")

        # Level 1 Predictions
        pred_lex = self.lexical_model.predict_proba(X_lex)[:, 1]
        pred_beh = self.behavioral_model.predict_proba(X_beh)[:, 1]
        pred_sem = self.semantic_model.predict_proba(X_sem)[:, 1]

        # Stack for Level 2
        X_level2 = np.column_stack([pred_lex, pred_beh, pred_sem])

        # Level 2 Prediction
        final_probs = self.meta_learner.predict_proba(X_level2)[:, 1]

        return final_probs
