import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

import library.config as config
import library.data_utils as data_utils
from library.models_lgbm import LGBMWrapper
from library.models_nn import NNWrapper


class StackingManager:
    """
    Manages the Heterogeneous Stacking Ensemble pipeline.
    Orchestrates data loading, K-Fold training of base models,
    meta-learner training, and submission generation.
    """

    def __init__(self):
        self.n_folds = config.N_FOLDS
        self.num_classes = config.NUM_CLASSES
        self.baseline_score = config.BASELINE_SCORE
        # Initialize Meta Learner (Level-1)
        self.meta_model = LogisticRegression(**config.META_PARAMS)

    def load_and_prepare_data(self):
        """
        Loads data using data_utils and combines train/val splits
        to allow for full Stratified K-Fold cross-validation.
        """
        # Load processed data (uses caching mechanism)
        data = data_utils.preprocess_data(load_cached_data=True)

        # Unpack Tree Data (Raw + Interactions)
        X_train_tree, y_train, X_val_tree, y_val, X_test_tree, test_ids = data["tree"]

        # Unpack NN Data (Scaled + Interactions)
        X_train_nn, _, X_val_nn, _, X_test_nn, _ = data["nn"]

        # Combine Train and Val for Stacking (Level-0 Training)
        # We reset index to ensure .iloc slicing works correctly during CV
        print("Combining Train and Validation sets for K-Fold Stacking...")
        X_full_tree = pd.concat([X_train_tree, X_val_tree], axis=0).reset_index(
            drop=True
        )
        X_full_nn = pd.concat([X_train_nn, X_val_nn], axis=0).reset_index(drop=True)
        y_full = np.concatenate([y_train, y_val], axis=0)

        return {
            "tree": (X_full_tree, X_test_tree),
            "nn": (X_full_nn, X_test_nn),
            "y": y_full,
            "test_ids": test_ids,
        }

    def cross_validate_base_models(self, data):
        """
        Trains Level-0 models (LightGBM & NN) using Stratified K-Fold.
        Generates OOF predictions (for meta-learner training) and
        Averaged Test predictions (for final inference).
        """
        X_tree, X_test_tree = data["tree"]
        X_nn, X_test_nn = data["nn"]
        y = data["y"]

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=config.SEED
        )

        # Storage for OOF predictions: Shape (N_samples, N_classes)
        oof_preds_lgbm = np.zeros((len(y), self.num_classes))
        oof_preds_nn = np.zeros((len(y), self.num_classes))

        # Storage for Test predictions (accumulate to average later)
        test_preds_lgbm_accum = np.zeros((len(X_test_tree), self.num_classes))
        test_preds_nn_accum = np.zeros((len(X_test_nn), self.num_classes))

        print(f"Starting {self.n_folds}-Fold Cross-Validation for Base Models...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tree, y)):
            print(f"\n--- Fold {fold + 1}/{self.n_folds} ---")

            # --- Prepare Fold Data ---
            # Tree Data
            X_tr_tree, X_va_tree = X_tree.iloc[train_idx], X_tree.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # NN Data
            X_tr_nn, X_va_nn = X_nn.iloc[train_idx], X_nn.iloc[val_idx]

            # --- Train LightGBM ---
            print("Training LightGBM...")
            lgbm = LGBMWrapper()
            lgbm.fit(X_tr_tree, y_tr, X_va_tree, y_va)

            # Predict OOF (Probabilities)
            oof_probs = lgbm.predict_proba(X_va_tree)
            oof_preds_lgbm[val_idx] = oof_probs

            # Predict Test (Probabilities)
            test_probs = lgbm.predict_proba(X_test_tree)
            test_preds_lgbm_accum += test_probs

            # --- Train Neural Network ---
            print("Training Neural Network...")
            nn_model = NNWrapper()
            nn_model.fit(X_tr_nn, y_tr, X_va_nn, y_va)

            # Predict OOF (Probabilities)
            oof_probs_nn = nn_model.predict_proba(X_va_nn)
            oof_preds_nn[val_idx] = oof_probs_nn

            # Predict Test (Probabilities)
            test_probs_nn = nn_model.predict_proba(X_test_nn)
            test_preds_nn_accum += test_probs_nn

        # Average Test Predictions across folds
        test_preds_lgbm_avg = test_preds_lgbm_accum / self.n_folds
        test_preds_nn_avg = test_preds_nn_accum / self.n_folds

        return oof_preds_lgbm, oof_preds_nn, test_preds_lgbm_avg, test_preds_nn_avg

    def train_meta_learner(self, X_meta, y):
        """
        Trains the Level-1 Meta Learner (Logistic Regression) on OOF probabilities.
        """
        print("\nTraining Meta Learner (Logistic Regression)...")
        self.meta_model.fit(X_meta, y)

        # Evaluate on training data (OOF) to get an estimate of ensemble performance
        preds = self.meta_model.predict(X_meta)
        acc = accuracy_score(y, preds)
        print(f"Meta Learner OOF Accuracy: {acc}")
        return acc

    def generate_submission(self, test_ids, predictions):
        """
        Maps predictions back to original class labels and saves the submission file.
        """
        # Map class indices (0-5) back to original labels (1, 2, 3, 4, 6, 7)
        inv_map = config.INV_CLASS_MAP
        final_labels = [inv_map[p] for p in predictions]

        sub_df = pd.DataFrame(
            {config.ID_COL: test_ids, config.TARGET_COL: final_labels}
        )

        print(f"Saving submission to {config.SUBMISSION_PATH}...")
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)

    def run(self):
        """
        Main execution pipeline.
        """
        # 1. Load Data
        data = self.load_and_prepare_data()
        y_full = data["y"]
        test_ids = data["test_ids"]

        # 2. Train Base Models & Get Predictions
        oof_lgbm, oof_nn, test_lgbm, test_nn = self.cross_validate_base_models(data)

        # 3. Construct Meta Features
        # Concatenate probabilities: [LGBM_Class0...LGBM_ClassN, NN_Class0...NN_ClassN]
        print("Constructing Meta-Features...")
        X_meta_oof = np.hstack([oof_lgbm, oof_nn])
        X_meta_test = np.hstack([test_lgbm, test_nn])

        # 4. Train Meta Learner
        oof_score = self.train_meta_learner(X_meta_oof, y_full)

        # 5. Champion-Challenger Guard
        print(f"\nEnsemble OOF Score: {oof_score}")
        print(f"Baseline Score: {self.baseline_score}")

        if oof_score > self.baseline_score:
            print("Ensemble outperformed baseline. Generating submission...")

            # Predict on Test Meta Features using the trained Meta Learner
            final_test_preds = self.meta_model.predict(X_meta_test)

            self.generate_submission(test_ids, final_test_preds)
        else:
            print("Ensemble failed to beat baseline. Aborting submission generation.")
