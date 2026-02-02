import numpy as np
import scipy.sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import N_FOLDS, RANDOM_SEED, XGB_EARLY_STOPPING_ROUNDS
from library.models import ModelFactory
from library.utils import time_execution, print_metric


class StackingTrainer:
    """
    Implements the Symmetric Dual-Topology Stacking Ensemble training pipeline.
    Handles Level 1 Cross-Validation, Level 2 Meta-Learning, and Validation-Guided Retraining.
    """

    def __init__(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Args:
            X_train_dict (dict): Dictionary containing training feature views.
                Keys: 'lexical_sparse', 'lexical_dense', 'behavioral_sparse', 'behavioral_dense', 'metadata'
            y_train (array-like): Training targets.
            X_val_dict (dict): Dictionary containing validation feature views (for final early stopping).
            y_val (array-like): Validation targets.
        """
        self.X_train_dict = X_train_dict
        self.y_train = y_train
        self.X_val_dict = X_val_dict
        self.y_val = y_val

        # Placeholders for trained models
        self.meta_learner = None
        self.final_base_models = {}

        # Define model keys for organization
        self.model_keys = [
            "lexical_sparse_rf",
            "lexical_dense_rf",
            "lexical_dense_xgb",
            "behavioral_sparse_rf",
            "behavioral_dense_xgb",
            "contextual_anchor_lr",
        ]

    def _prepare_input(self, view_name, X_dict, is_sparse=False):
        """
        Helper to concatenate the specific view features with the Unified Metadata Vector.

        Args:
            view_name (str): Key for the feature view in X_dict.
            X_dict (dict): Dictionary of feature arrays.
            is_sparse (bool): Whether the view is sparse (requires scipy.sparse.hstack).

        Returns:
            Combined feature matrix (CSR sparse or Numpy array).
        """
        # Contextual Anchor uses ONLY metadata
        if view_name == "metadata":
            return X_dict["metadata"]

        features = X_dict[view_name]
        metadata = X_dict["metadata"]

        if is_sparse:
            # Stack sparse features with dense metadata
            # Convert metadata to sparse CSR then hstack for efficiency
            meta_sparse = scipy.sparse.csr_matrix(metadata)
            combined = scipy.sparse.hstack([features, meta_sparse])
            return combined.tocsr()
        else:
            # Stack dense features with dense metadata
            return np.hstack([features, metadata])

    def _get_model_input(self, model_key, X_dict):
        """
        Routes the correct feature combination to the model based on its key.
        """
        if model_key == "lexical_sparse_rf":
            return self._prepare_input("lexical_sparse", X_dict, is_sparse=True)
        elif model_key == "lexical_dense_rf":
            return self._prepare_input("lexical_dense", X_dict, is_sparse=False)
        elif model_key == "lexical_dense_xgb":
            return self._prepare_input("lexical_dense", X_dict, is_sparse=False)
        elif model_key == "behavioral_sparse_rf":
            return self._prepare_input("behavioral_sparse", X_dict, is_sparse=True)
        elif model_key == "behavioral_dense_xgb":
            return self._prepare_input("behavioral_dense", X_dict, is_sparse=False)
        elif model_key == "contextual_anchor_lr":
            return self._prepare_input("metadata", X_dict, is_sparse=False)
        else:
            raise ValueError(f"Unknown model key: {model_key}")

    def _get_new_base_model(self, model_key):
        """Instantiates a fresh base learner from ModelFactory."""
        if model_key == "lexical_sparse_rf":
            return ModelFactory.get_lexical_sparse_rf()
        elif model_key == "lexical_dense_rf":
            return ModelFactory.get_lexical_dense_rf()
        elif model_key == "lexical_dense_xgb":
            return ModelFactory.get_lexical_dense_xgb()
        elif model_key == "behavioral_sparse_rf":
            return ModelFactory.get_behavioral_sparse_rf()
        elif model_key == "behavioral_dense_xgb":
            return ModelFactory.get_behavioral_dense_xgb()
        elif model_key == "contextual_anchor_lr":
            return ModelFactory.get_contextual_anchor_lr()
        else:
            raise ValueError(f"Unknown model key: {model_key}")

    @time_execution
    def run_cv_and_meta_training(self):
        """
        Performs Level 1 Stacking (Cross-Validation) to generate OOF predictions
        and trains the Level 2 Meta-Learner.
        """
        print(f"Starting Level 1 Cross-Validation ({N_FOLDS} folds)...")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        # Initialize OOF matrix: (n_samples, n_models)
        n_samples = len(self.y_train)
        oof_preds = np.zeros((n_samples, len(self.model_keys)))

        # Pre-compute inputs for the whole training set to avoid re-stacking inside loop
        X_train_prepared = {
            key: self._get_model_input(key, self.X_train_dict)
            for key in self.model_keys
        }

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), self.y_train)
        ):
            y_fold_train = self.y_train[train_idx]
            y_fold_val = self.y_train[val_idx]

            for col_idx, key in enumerate(self.model_keys):
                model = self._get_new_base_model(key)
                X_fold_train = X_train_prepared[key][train_idx]
                X_fold_val = X_train_prepared[key][val_idx]

                # Handle XGBoost Early Stopping within CV
                if "xgb" in key:
                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    )
                else:
                    model.fit(X_fold_train, y_fold_train)

                # Predict probabilities (positive class)
                preds = model.predict_proba(X_fold_val)[:, 1]
                oof_preds[val_idx, col_idx] = preds

        # Calculate and print OOF AUC for each base model
        print("\nLevel 1 OOF Performance:")
        for col_idx, key in enumerate(self.model_keys):
            auc = roc_auc_score(self.y_train, oof_preds[:, col_idx])
            print_metric(f"  {key} AUC", auc)

        # Train Meta-Learner
        print("\nTraining Level 2 Meta-Learner...")
        self.meta_learner = ModelFactory.get_meta_learner()
        self.meta_learner.fit(oof_preds, self.y_train)

        # Meta-learner OOF Score
        meta_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(self.y_train, meta_preds)
        print_metric("  Meta-Learner OOF AUC", meta_auc)

        # Print Meta-Learner Coefficients
        print("\nMeta-Learner Coefficients:")
        for key, coef in zip(self.model_keys, self.meta_learner.coef_[0]):
            print(f"  {key}: {coef:.4f}")

    @time_execution
    def train_final_base_models(self):
        """
        Retrains all base models for final inference.
        Strategy:
        - RF/LR: Train on concatenated Train + Val sets.
        - XGB: Train on Train set, use Val set for Early Stopping.
        """
        print("\nRetraining Final Base Models (Validation-Guided)...")

        # Prepare inputs
        X_train_prepared = {
            key: self._get_model_input(key, self.X_train_dict)
            for key in self.model_keys
        }
        X_val_prepared = {
            key: self._get_model_input(key, self.X_val_dict) for key in self.model_keys
        }

        for key in self.model_keys:
            model = self._get_new_base_model(key)

            if "xgb" in key:
                # XGBoost: Train on Train, Early Stop on Val
                model.fit(
                    X_train_prepared[key],
                    self.y_train,
                    eval_set=[(X_val_prepared[key], self.y_val)],
                    early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            else:
                # RF/LR: Train on Train + Val
                # Concatenate data
                X_tr = X_train_prepared[key]
                X_vl = X_val_prepared[key]

                if scipy.sparse.issparse(X_tr):
                    X_full = scipy.sparse.vstack([X_tr, X_vl])
                else:
                    X_full = np.concatenate([X_tr, X_vl], axis=0)

                y_full = np.concatenate([self.y_train, self.y_val], axis=0)

                model.fit(X_full, y_full)

            self.final_base_models[key] = model
            print(f"  {key} retrained.")

    @time_execution
    def predict(self, X_test_dict):
        """
        Generates final predictions for the test set.

        Args:
            X_test_dict (dict): Dictionary containing test feature views.

        Returns:
            np.array: Probability of success (class 1).
        """
        print("\nGenerating Final Predictions...")

        # 1. Generate Level 1 Predictions
        n_samples = X_test_dict["metadata"].shape[0]
        l1_preds = np.zeros((n_samples, len(self.model_keys)))

        for col_idx, key in enumerate(self.model_keys):
            model = self.final_base_models[key]
            X_input = self._get_model_input(key, X_test_dict)
            l1_preds[:, col_idx] = model.predict_proba(X_input)[:, 1]

        # 2. Generate Level 2 Predictions (Meta-Learner)
        final_probs = self.meta_learner.predict_proba(l1_preds)[:, 1]

        return final_probs
