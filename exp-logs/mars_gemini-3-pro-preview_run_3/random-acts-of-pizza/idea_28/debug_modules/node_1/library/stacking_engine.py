import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import Timer, set_seed
from library.model_factory import (
    get_lexical_bagger,
    get_community_bagger,
    get_semantic_booster,
    get_semantic_bagger,
    get_metadata_anchor,
    get_meta_learner,
)


class NestedStackingTrainer:
    """
    Implements the Robust Pent-View Stacking Ensemble with Nested Internal Validation.
    Orchestrates Level 1 base learner training, OOF generation, and Level 2 meta-learning.
    """

    def __init__(self, train_features: dict, y_train: pd.Series):
        """
        Args:
            train_features (dict): Dictionary of feature matrices (lexical, behavioral, semantic, metadata).
            y_train (pd.Series): Target labels.
        """
        self.train_features = train_features
        # Ensure y_train is a Series with reset index for proper slicing
        self.y_train = (
            y_train.reset_index(drop=True)
            if isinstance(y_train, pd.Series)
            else pd.Series(y_train)
        )

        # Storage for retrained models
        self.final_models = {}
        self.meta_learner = None
        self.avg_xgb_best_iteration = None

    def _prepare_data(self, features: dict, indices: np.ndarray = None):
        """
        Slices and concatenates features for specific views.
        Concatenates Global Metadata to all views as per architecture.
        """
        # Extract raw views
        lex = features["lexical"]
        beh = features["behavioral"]
        sem = features["semantic"]
        meta = features["metadata"]

        # Slice if indices provided
        if indices is not None:
            lex = lex[indices]
            beh = beh[indices]
            sem = sem[indices]
            meta = meta[indices]

        # Convert metadata to sparse for sparse concatenation where necessary
        meta_sparse = sparse.csr_matrix(meta)

        # 1. Lexical View: Sparse Text + Dense Metadata -> Sparse
        X_lex = sparse.hstack([lex, meta_sparse]).tocsr()

        # 2. Behavioral View: Sparse History + Dense Metadata -> Sparse
        X_beh = sparse.hstack([beh, meta_sparse]).tocsr()

        # 3. Semantic View: Dense Embeddings + Dense Metadata -> Dense
        X_sem = np.hstack([sem, meta])

        # 4. Metadata View: Dense Metadata -> Dense
        X_meta = meta

        return X_lex, X_beh, X_sem, X_meta

    def train_cv(self) -> np.ndarray:
        """
        Performs 5-Fold Stratified CV to generate OOF predictions.
        Implements Nested Internal Validation for XGBoost.
        """
        set_seed(Config.SEED)

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        n_samples = len(self.y_train)
        # 5 Base Learners
        oof_preds = np.zeros((n_samples, 5))
        xgb_best_iters = []

        print(f"Starting {Config.N_FOLDS}-Fold Stacking CV...")

        fold = 0
        for train_idx, val_idx in skf.split(np.zeros(n_samples), self.y_train):
            fold += 1

            # Prepare Data
            X_lex_tr, X_beh_tr, X_sem_tr, X_meta_tr = self._prepare_data(
                self.train_features, train_idx
            )
            X_lex_val, X_beh_val, X_sem_val, X_meta_val = self._prepare_data(
                self.train_features, val_idx
            )

            y_tr_fold = self.y_train.iloc[train_idx]

            # --- 1. Lexical Bagger (RF) ---
            model_lex = get_lexical_bagger()
            model_lex.fit(X_lex_tr, y_tr_fold)
            oof_preds[val_idx, 0] = model_lex.predict_proba(X_lex_val)[:, 1]

            # --- 2. Community Bagger (RF) ---
            model_comm = get_community_bagger()
            model_comm.fit(X_beh_tr, y_tr_fold)
            oof_preds[val_idx, 1] = model_comm.predict_proba(X_beh_val)[:, 1]

            # --- 3. Semantic Bagger (RF) ---
            model_sem_rf = get_semantic_bagger()
            model_sem_rf.fit(X_sem_tr, y_tr_fold)
            oof_preds[val_idx, 2] = model_sem_rf.predict_proba(X_sem_val)[:, 1]

            # --- 4. Metadata Anchor (LR) ---
            model_meta = get_metadata_anchor()
            model_meta.fit(X_meta_tr, y_tr_fold)
            oof_preds[val_idx, 3] = model_meta.predict_proba(X_meta_val)[:, 1]

            # --- 5. Semantic Booster (XGB) with Nested Validation ---
            # Internal Split for Early Stopping
            X_sem_inner_tr, X_sem_inner_val, y_inner_tr, y_inner_val = train_test_split(
                X_sem_tr,
                y_tr_fold,
                test_size=Config.INTERNAL_VAL_SIZE,
                stratify=y_tr_fold,
                random_state=Config.SEED,
            )

            model_xgb = get_semantic_booster()
            model_xgb.set_params(early_stopping_rounds=Config.XGB_EARLY_STOPPING_ROUNDS)
            model_xgb.fit(
                X_sem_inner_tr,
                y_inner_tr,
                eval_set=[(X_sem_inner_val, y_inner_val)],
                verbose=False,
            )

            # Capture best iteration (1-based for n_estimators)
            best_iter = model_xgb.best_iteration + 1
            xgb_best_iters.append(best_iter)

            # Predict on Outer Validation Fold
            oof_preds[val_idx, 4] = model_xgb.predict_proba(X_sem_val)[:, 1]

        # Calculate Average Best Iteration
        self.avg_xgb_best_iteration = int(np.mean(xgb_best_iters))
        print(f"Average XGBoost Optimal Iterations: {self.avg_xgb_best_iteration}")

        # Calculate and Print OOF AUC (Simple Average Proxy)
        avg_oof = np.mean(oof_preds, axis=1)
        auc = roc_auc_score(self.y_train, avg_oof)
        print(f"Level 1 OOF AUC (Simple Average): {auc}")

        return oof_preds

    def train_meta_learner(self, oof_preds: np.ndarray):
        """
        Trains the Level 2 Logistic Regression on OOF predictions.
        """
        print("Training Meta-Learner...")
        self.meta_learner = get_meta_learner()
        self.meta_learner.fit(oof_preds, self.y_train)

        # Validate on OOF (sanity check)
        meta_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        auc = roc_auc_score(self.y_train, meta_preds)
        print(f"Level 2 Stacked AUC (OOF Re-eval): {auc}")

    def retrain_full_models(self):
        """
        Retrains all Level 1 models on the full training dataset.
        Uses the average optimal iterations for XGBoost.
        """
        print("Retraining Level 1 Models on Full Data...")

        X_lex, X_beh, X_sem, X_meta = self._prepare_data(self.train_features)
        y = self.y_train

        # 1. Lexical Bagger
        self.final_models["lexical_bagger"] = get_lexical_bagger().fit(X_lex, y)

        # 2. Community Bagger
        self.final_models["community_bagger"] = get_community_bagger().fit(X_beh, y)

        # 3. Semantic Bagger
        self.final_models["semantic_bagger"] = get_semantic_bagger().fit(X_sem, y)

        # 4. Metadata Anchor
        self.final_models["metadata_anchor"] = get_metadata_anchor().fit(X_meta, y)

        # 5. Semantic Booster
        # Update n_estimators to average optimal from CV
        model_xgb = get_semantic_booster()
        model_xgb.set_params(n_estimators=self.avg_xgb_best_iteration)
        # Train without early stopping
        self.final_models["semantic_booster"] = model_xgb.fit(X_sem, y, verbose=False)

    def predict_ensemble(self, test_features: dict) -> np.ndarray:
        """
        Generates final predictions for the test set.
        """
        X_lex, X_beh, X_sem, X_meta = self._prepare_data(test_features)
        n_samples = X_meta.shape[0]

        # Matrix for Level 1 predictions
        l1_preds = np.zeros((n_samples, 5))

        l1_preds[:, 0] = self.final_models["lexical_bagger"].predict_proba(X_lex)[:, 1]
        l1_preds[:, 1] = self.final_models["community_bagger"].predict_proba(X_beh)[
            :, 1
        ]
        l1_preds[:, 2] = self.final_models["semantic_bagger"].predict_proba(X_sem)[:, 1]
        l1_preds[:, 3] = self.final_models["metadata_anchor"].predict_proba(X_meta)[
            :, 1
        ]
        l1_preds[:, 4] = self.final_models["semantic_booster"].predict_proba(X_sem)[
            :, 1
        ]

        # Level 2 Prediction
        final_probs = self.meta_learner.predict_proba(l1_preds)[:, 1]

        return final_probs
