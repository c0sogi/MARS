import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

import library.config as config
from library.utils import timer, calculate_auc, set_seed, save_submission
from library.data_loader import load_data
from library.feature_pipeline import generate_features


class TriViewStackingEnsemble:
    """
    Implements a 2-Level Stacking Ensemble with three distinct views:
    1. Lexical View (Sparse TF-IDF) -> Random Forest
    2. Semantic View (Dense SBERT) -> Random Forest
    3. Community View (Dense SVD) -> XGBoost

    Level 2: Logistic Regression Meta-Learner
    """

    def __init__(self):
        set_seed(config.SEED)

        # Initialize Base Learners (Level 1)
        self.lexical_model = RandomForestClassifier(**config.L1_LEXICAL_PARAMS)
        self.semantic_model = RandomForestClassifier(**config.L1_SEMANTIC_PARAMS)
        self.community_model = xgb.XGBClassifier(**config.L1_COMMUNITY_PARAMS)

        # Initialize Meta Learner (Level 2)
        self.meta_model = LogisticRegression(**config.L2_META_PARAMS)

        # Placeholders for retrained models
        self.final_lexical = None
        self.final_semantic = None
        self.final_community = None
        self.final_meta = None

    def _get_xgb_params(self, y_train):
        """Calculates dynamic parameters for XGBoost (e.g., scale_pos_weight)."""
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos
        params = config.L1_COMMUNITY_PARAMS.copy()
        params["scale_pos_weight"] = scale_pos_weight
        return params

    def fit(self, X_train_dict, y_train):
        """
        Orchestrates the Stacking process:
        1. 5-Fold CV to generate OOF predictions.
        2. Train Meta-Learner on OOFs.
        3. Retrain Base Learners on full data.
        """
        print(f"Starting Stacking Ensemble Training with {config.N_FOLDS} folds...")

        # 1. Generate Out-Of-Fold (OOF) Predictions
        oof_preds, meta_y = self._generate_oof(X_train_dict, y_train)

        # 2. Train Meta-Learner
        print("Training Level 2 Meta-Learner...")
        self.meta_model.fit(oof_preds, meta_y)
        self.final_meta = self.meta_model

        # Evaluate Meta-Learner on OOF (Approximation of generalization error)
        meta_oof_probs = self.meta_model.predict_proba(oof_preds)[:, 1]
        calculate_auc(meta_y, meta_oof_probs, label="Level 2 OOF Stacking")

        # 3. Retrain Base Learners on Full Data
        print("Retraining Level 1 Base Learners on full dataset...")
        self._retrain_full(X_train_dict, y_train)

        return self

    def _generate_oof(self, X_dict, y):
        """Performs Stratified K-Fold CV to generate OOF predictions."""
        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        n_samples = len(y)
        oof_lexical = np.zeros(n_samples)
        oof_semantic = np.zeros(n_samples)
        oof_community = np.zeros(n_samples)

        # Ensure y is numpy array for indexing
        y_arr = np.array(y)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_arr)
        ):
            # print(f"  Processing Fold {fold + 1}/{N_FOLDS}...")

            # Slice Data
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # --- Lexical Branch (RF) ---
            X_lex_tr = X_dict["lexical"][train_idx]
            X_lex_val = X_dict["lexical"][val_idx]

            model_lex = clone(self.lexical_model)
            model_lex.fit(X_lex_tr, y_tr)
            oof_lexical[val_idx] = model_lex.predict_proba(X_lex_val)[:, 1]

            # --- Semantic Branch (RF) ---
            X_sem_tr = X_dict["semantic"][train_idx]
            X_sem_val = X_dict["semantic"][val_idx]

            model_sem = clone(self.semantic_model)
            model_sem.fit(X_sem_tr, y_tr)
            oof_semantic[val_idx] = model_sem.predict_proba(X_sem_val)[:, 1]

            # --- Community Branch (XGB) ---
            X_com_tr = X_dict["community"][train_idx]
            X_com_val = X_dict["community"][val_idx]

            # Update scale_pos_weight for this fold
            fold_xgb_params = self._get_xgb_params(y_tr)
            model_com = xgb.XGBClassifier(**fold_xgb_params)

            model_com.fit(X_com_tr, y_tr, eval_set=[(X_com_val, y_val)], verbose=False)
            oof_community[val_idx] = model_com.predict_proba(X_com_val)[:, 1]

        # Stack OOF predictions
        oof_matrix = np.column_stack([oof_lexical, oof_semantic, oof_community])
        return oof_matrix, y_arr

    def _retrain_full(self, X_dict, y):
        """Retrains base models on the complete training set."""
        y_arr = np.array(y)

        # Lexical
        self.final_lexical = clone(self.lexical_model)
        self.final_lexical.fit(X_dict["lexical"], y_arr)

        # Semantic
        self.final_semantic = clone(self.semantic_model)
        self.final_semantic.fit(X_dict["semantic"], y_arr)

        # Community
        xgb_params = self._get_xgb_params(y_arr)
        self.final_community = xgb.XGBClassifier(**xgb_params)
        self.final_community.fit(X_dict["community"], y_arr, verbose=False)

    def predict_proba(self, X_dict):
        """
        Generates final predictions using the stacked ensemble.
        """
        if not self.final_meta:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Level 1 Predictions
        p_lexical = self.final_lexical.predict_proba(X_dict["lexical"])[:, 1]
        p_semantic = self.final_semantic.predict_proba(X_dict["semantic"])[:, 1]
        p_community = self.final_community.predict_proba(X_dict["community"])[:, 1]

        # Stack
        L1_preds = np.column_stack([p_lexical, p_semantic, p_community])

        # Level 2 Prediction
        final_probs = self.final_meta.predict_proba(L1_preds)[:, 1]

        return final_probs


def run_stacking_pipeline(load_cached_data=True, debug=False):
    """
    Main driver function to execute the stacking pipeline.
    """
    set_seed(config.SEED)

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Generate Features (Lexical, Semantic, Community)
    # Note: We combine Train and Val for the final fit in a real competition setting,
    # but here we keep them separate to validate the stacking logic first,
    # or we can merge them if we trust the OOF CV score.
    # Given the prompt implies using X_val for validation, we will generate features for all.
    train_feats, val_feats, test_feats = generate_features(
        X_train, X_val, X_test, load_cached_data=load_cached_data
    )

    # 3. Instantiate and Train Ensemble
    ensemble = TriViewStackingEnsemble()

    # We fit on X_train (which is the training split from metadata)
    # The ensemble performs internal CV on this data.
    with timer("Ensemble Training"):
        ensemble.fit(train_feats, y_train)

    # 4. Validation Evaluation (Separate Hold-out Set)
    # This evaluates how well the retrained L1 models + L2 meta learner generalize
    # to the explicit validation set provided in metadata.
    print("\nEvaluating on Hold-out Validation Set...")
    val_probs = ensemble.predict_proba(val_feats)
    calculate_auc(y_val, val_probs, label="Hold-out Validation")

    # 5. Test Prediction & Submission
    print("\nGenerating Test Predictions...")
    test_probs = ensemble.predict_proba(test_feats)

    save_submission(test_ids, test_probs, path=config.SUBMISSION_PATH)

    return ensemble
