import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config


class Trainer:
    def __init__(self):
        self.models_dir = os.path.join(Config.CACHE_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_SEED

    def _concat_features(self, X_main, X_meta):
        """
        Concatenates the main modality features with the metadata features.
        Handles both sparse and dense main features.
        """
        # If X_main is None (e.g. Metadata branch where X_main is just X_meta),
        # we might just return X_meta, but the architecture implies
        # specific branches.
        # Case 1: Metadata branch. X_main might be None or same as X_meta.
        if X_main is None:
            return X_meta

        # Case 2: Sparse X_main (Lexical, Behavioral)
        if sparse.issparse(X_main):
            # Ensure X_meta is sparse for efficient hstack
            X_meta_sparse = sparse.csr_matrix(X_meta)
            return sparse.hstack([X_main, X_meta_sparse], format="csr")

        # Case 3: Dense X_main (Semantic)
        else:
            return np.hstack([X_main, X_meta])

    def _get_model_path(self, model_name, fold=None):
        if fold is None:
            return os.path.join(self.models_dir, f"{model_name}.joblib")
        else:
            return os.path.join(self.models_dir, f"{model_name}_fold_{fold}.joblib")

    def train_cv_volatile(self, model_name, model_factory_func, X_main, X_meta, y):
        """
        Trains volatile models (XGB, LGBM) using CV with Early Stopping.
        Saves a model for EACH fold.
        Returns OOF predictions.
        """
        print(f"\nStarting Volatile CV Training for: {model_name}")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        oof_preds = np.zeros(len(y))

        # Determine early stopping rounds from Config based on model name
        # This is a heuristic based on the naming convention in Config
        es_rounds = 50  # Default
        if "semantic_booster" in model_name:
            es_rounds = Config.SEMANTIC_BOOSTER_PARAMS.get("early_stopping_rounds", 100)
        elif "semantic_gradient" in model_name:
            es_rounds = Config.SEMANTIC_GRADIENT_PARAMS.get(
                "early_stopping_rounds", 100
            )
        elif "temporal_booster" in model_name:
            es_rounds = Config.TEMPORAL_BOOSTER_PARAMS.get("early_stopping_rounds", 50)

        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            # 1. Split Data
            # Note: We split indices first, then concatenate features to save memory
            # (avoiding creating full concatenated matrix if possible, though _concat_features does it)
            # For efficiency with sparse matrices, it's often better to slice then stack
            # or stack then slice. Here we stack then slice to ensure consistency,
            # but slicing sparse matrices is fast.

            # Construct full feature matrix for this fold's usage
            # To save memory, we can slice inputs first then concat
            X_main_train = X_main[train_idx] if X_main is not None else None
            X_main_val = X_main[val_idx] if X_main is not None else None

            X_meta_train = X_meta[train_idx]
            X_meta_val = X_meta[val_idx]

            X_train_fold = self._concat_features(X_main_train, X_meta_train)
            X_val_fold = self._concat_features(X_main_val, X_meta_val)
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # 2. Instantiate new model for this fold
            model = model_factory_func()

            # 3. Fit with Early Stopping
            # XGBoost and LightGBM sklearn API
            fit_params = {"eval_set": [(X_val_fold, y_val_fold)]}

            # XGBoost supports verbose in fit to suppress logging
            # LightGBM 4.0+ removed verbose from fit (handled in init or callbacks)
            if "XGB" in model.__class__.__name__:
                fit_params["verbose"] = False

            model.fit(X_train_fold, y_train_fold, **fit_params)

            # 4. Predict
            # Best iteration is automatically used if early_stopping was active and load_best_model_at_end=True (default in many)
            val_preds = model.predict_proba(X_val_fold)[:, 1]
            oof_preds[val_idx] = val_preds

            # 5. Score
            fold_score = roc_auc_score(y_val_fold, val_preds)
            scores.append(fold_score)
            print(f"  Fold {fold} AUC: {fold_score:.16f}")

            # 6. Save Fold Model
            save_path = self._get_model_path(model_name, fold)
            joblib.dump(model, save_path)

        avg_score = np.mean(scores)
        print(f"{model_name} CV Average AUC: {avg_score:.16f}")
        return oof_preds

    def train_cv_stable(self, model_name, model_factory_func, X_main, X_meta, y):
        """
        Trains stable models (RF, Linear) using CV.
        Generates OOF predictions.
        Saves fold models (for consistency/debugging).
        """
        print(f"\nStarting Stable CV Training for: {model_name}")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        oof_preds = np.zeros(len(y))
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            # Slice then concat
            X_main_train = X_main[train_idx] if X_main is not None else None
            X_main_val = X_main[val_idx] if X_main is not None else None

            X_meta_train = X_meta[train_idx]
            X_meta_val = X_meta[val_idx]

            X_train_fold = self._concat_features(X_main_train, X_meta_train)
            X_val_fold = self._concat_features(X_main_val, X_meta_val)
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # Instantiate
            model = model_factory_func()

            # Fit
            model.fit(X_train_fold, y_train_fold)

            # Predict
            val_preds = model.predict_proba(X_val_fold)[:, 1]
            oof_preds[val_idx] = val_preds

            # Score
            fold_score = roc_auc_score(y_val_fold, val_preds)
            scores.append(fold_score)
            print(f"  Fold {fold} AUC: {fold_score:.16f}")

            # Save Fold Model
            save_path = self._get_model_path(model_name, fold)
            joblib.dump(model, save_path)

        avg_score = np.mean(scores)
        print(f"{model_name} CV Average AUC: {avg_score:.16f}")
        return oof_preds

    def train_full_stable(self, model_name, model_factory_func, X_main, X_meta, y):
        """
        Retrains a stable model on the FULL dataset.
        Saves the single model instance.
        """
        print(f"\nRetraining Full Stable Model: {model_name}")

        # Concat full dataset
        X_full = self._concat_features(X_main, X_meta)

        # Instantiate
        model = model_factory_func()

        # Fit
        model.fit(X_full, y)

        # Save (no fold suffix)
        save_path = self._get_model_path(model_name)
        joblib.dump(model, save_path)
        print(f"Saved full model to {save_path}")

        return model

    def train_meta_learner(self, meta_model, X_oof, y):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        print("\nTraining Level 2 Meta-Learner...")

        # Simple CV check for meta learner performance
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        scores = []
        meta_oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y)):
            X_tr, X_val = X_oof[train_idx], X_oof[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf = clone(meta_model)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_val)[:, 1]
            meta_oof_preds[val_idx] = preds
            score = roc_auc_score(y_val, preds)
            scores.append(score)
            print(f"  Meta Fold {fold} AUC: {score:.16f}")

        print(f"Meta-Learner CV Average AUC: {np.mean(scores):.16f}")

        # Fit on full OOF
        meta_model.fit(X_oof, y)
        save_path = self._get_model_path("meta_learner")
        joblib.dump(meta_model, save_path)
        print(f"Saved Meta-Learner to {save_path}")

        return meta_model, meta_oof_preds
