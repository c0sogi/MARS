import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import MODELS_DIR
from library.utils import Timer, save_joblib, load_joblib
from library.model_factory import get_base_models, get_meta_learner


class StackingEngine:
    """
    Implements the Hept-View Stacking Ensemble training and inference engine.
    Manages 5-fold CV, hybrid inference protocols (averaging vs. refitting),
    and Level 2 meta-learning.
    """

    def __init__(self):
        self.models_dir = MODELS_DIR
        # Define feature mapping based on model branches
        self.feature_mapping = {
            # Lexical Branch: Sparse Text + Dense Meta
            "lexical_bagger": ["lexical", "meta"],
            # Behavioral Branch: Sparse History + Dense Meta
            "community_bagger": ["behavioral", "meta"],
            # Semantic Branch: Dense Embeddings + Dense Meta
            "semantic_booster": ["semantic", "meta"],
            "semantic_gradient": ["semantic", "meta"],
            "semantic_bagger": ["semantic", "meta"],
            # Contextual Branch: Dense Meta only
            "metadata_anchor": ["meta"],
            "temporal_booster": ["meta"],
        }

    def _get_model_features(self, X_dict, model_name):
        """
        Constructs the specific feature matrix for a given model by concatenating
        the required modalities. Handles sparse/dense combinations.
        """
        required_keys = self.feature_mapping.get(model_name)
        if not required_keys:
            raise ValueError(f"No feature mapping defined for {model_name}")

        components = [X_dict[key] for key in required_keys]

        # Check if any component is sparse
        is_sparse = any(sparse.issparse(c) for c in components)

        if is_sparse:
            # Use scipy.sparse.hstack for mixed or all-sparse data
            return sparse.hstack(components, format="csr")
        else:
            # Use numpy.hstack for all-dense data
            return np.hstack(components)

    def train(self, X_dict, y, folds):
        """
        Executes the Hybrid Training Protocol:
        1. 5-Fold CV to generate OOF predictions.
        2. Volatile models use Early Stopping per fold.
        3. Stable models are trained per fold for OOF.
        4. Meta-learner trained on OOF matrix.
        5. Stable models retrained on full dataset.

        Args:
            X_dict (dict): Dictionary of feature matrices.
            y (array-like): Target vector.
            folds (list): List of (train_idx, val_idx) tuples.
        """
        # Get fresh model instances
        base_models_struct = get_base_models()
        volatile_models = base_models_struct["volatile"]
        stable_models = base_models_struct["stable"]

        # Flatten model list for OOF storage
        all_model_names = list(volatile_models.keys()) + list(stable_models.keys())
        oof_preds = pd.DataFrame(index=np.arange(len(y)), columns=all_model_names)
        oof_preds[:] = np.nan

        print(f"Starting training with {len(folds)} folds...")

        # --- Step 1: Cross-Validation Loop ---
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{len(folds)} ---")

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 1.1 Train Volatile Models (with Early Stopping)
            for name, model in volatile_models.items():
                # Prepare features
                X_combined = self._get_model_features(X_dict, name)
                X_train_fold = X_combined[train_idx]
                X_val_fold = X_combined[val_idx]

                # Clone to ensure fresh start
                clf = clone(model)

                # Fit with Early Stopping
                # Note: XGBClassifier and LGBMClassifier in config have early_stopping params set.
                # We pass eval_set to trigger it.
                clf.fit(X_train_fold, y_train, eval_set=[(X_val_fold, y_val)])

                # Predict OOF
                val_preds = clf.predict_proba(X_val_fold)[:, 1]
                oof_preds.loc[val_idx, name] = val_preds

                # Save Fold Model
                save_path = os.path.join(
                    self.models_dir, f"{name}_fold_{fold_idx}.joblib"
                )
                save_joblib(clf, save_path)

                score = roc_auc_score(y_val, val_preds)
                print(f"  {name}: AUC = {score:.8f}")

            # 1.2 Train Stable Models (Standard Fit)
            for name, model in stable_models.items():
                # Prepare features
                X_combined = self._get_model_features(X_dict, name)
                X_train_fold = X_combined[train_idx]
                X_val_fold = X_combined[val_idx]

                # Clone
                clf = clone(model)

                # Standard Fit
                clf.fit(X_train_fold, y_train)

                # Predict OOF
                val_preds = clf.predict_proba(X_val_fold)[:, 1]
                oof_preds.loc[val_idx, name] = val_preds

                # Save Fold Model
                save_path = os.path.join(
                    self.models_dir, f"{name}_fold_{fold_idx}.joblib"
                )
                save_joblib(clf, save_path)

                score = roc_auc_score(y_val, val_preds)
                print(f"  {name}: AUC = {score:.8f}")

        # --- Step 2: Train Meta-Learner ---
        print("\n--- Training Meta-Learner ---")
        meta_learner = get_meta_learner()

        # Check for NaNs in OOF (should not happen if CV is correct)
        if oof_preds.isnull().any().any():
            print("Warning: NaNs found in OOF predictions. Filling with 0.5")
            oof_preds = oof_preds.fillna(0.5)

        meta_learner.fit(oof_preds, y)

        # Evaluate Meta-Learner on OOF
        meta_oof_preds = meta_learner.predict_proba(oof_preds)[:, 1]
        meta_score = roc_auc_score(y, meta_oof_preds)
        print(f"Meta-Learner OOF AUC: {meta_score:.8f}")

        save_joblib(meta_learner, os.path.join(self.models_dir, "meta_learner.joblib"))

        # --- Step 3: Retrain Stable Models on Full Data ---
        print("\n--- Retraining Stable Models on Full Dataset ---")
        for name, model in stable_models.items():
            X_all = self._get_model_features(X_dict, name)

            clf = clone(model)
            clf.fit(X_all, y)

            save_path = os.path.join(self.models_dir, f"{name}_full.joblib")
            save_joblib(clf, save_path)
            print(f"  {name} retrained and saved.")

    def predict(self, X_dict):
        """
        Executes Hybrid Inference:
        1. Volatile Models: Average predictions from 5 saved fold-models.
        2. Stable Models: Use the single fully-retrained model.
        3. Meta-Learner: Combine predictions.

        Args:
            X_dict (dict): Dictionary of feature matrices for test set.

        Returns:
            np.array: Final probabilities.
        """
        # Re-fetch structure to know which is which
        base_models_struct = get_base_models()
        volatile_models = base_models_struct["volatile"]
        stable_models = base_models_struct["stable"]

        all_model_names = list(volatile_models.keys()) + list(stable_models.keys())
        n_samples = X_dict["meta"].shape[0]

        # Level 1 Predictions Matrix
        l1_preds = pd.DataFrame(index=np.arange(n_samples), columns=all_model_names)

        print("Generating Level 1 Predictions...")

        # 1. Volatile Models: CV-Bagging (Average 5 folds)
        for name in volatile_models.keys():
            X_test = self._get_model_features(X_dict, name)
            fold_preds = []

            # Load each fold model
            # Assuming 5 folds as per config N_FOLDS, but we check file existence or loop range
            # We'll assume standard 5 folds here as defined in config, or check directory
            from library.config import N_FOLDS

            for i in range(N_FOLDS):
                model_path = os.path.join(self.models_dir, f"{name}_fold_{i}.joblib")
                model = load_joblib(model_path)
                pred = model.predict_proba(X_test)[:, 1]
                fold_preds.append(pred)

            # Average
            avg_pred = np.mean(fold_preds, axis=0)
            l1_preds[name] = avg_pred

        # 2. Stable Models: Use Full Model
        for name in stable_models.keys():
            X_test = self._get_model_features(X_dict, name)
            model_path = os.path.join(self.models_dir, f"{name}_full.joblib")
            model = load_joblib(model_path)
            pred = model.predict_proba(X_test)[:, 1]
            l1_preds[name] = pred

        # 3. Meta-Learner Prediction
        print("Generating Final Predictions...")
        meta_learner_path = os.path.join(self.models_dir, "meta_learner.joblib")
        meta_learner = load_joblib(meta_learner_path)

        final_preds = meta_learner.predict_proba(l1_preds)[:, 1]

        return final_preds
