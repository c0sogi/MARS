import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from library.config import (
    N_FOLDS,
    RANDOM_SEED,
    XGB_EARLY_STOPPING_ROUNDS,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
)
from library.model_definitions import (
    get_lexical_rf,
    get_behavioral_rf,
    get_semantic_xgb,
    get_semantic_rf,
    get_contextual_lr,
    get_meta_learner,
)
from library.utils import timer, print_header


class PentViewStackingEnsemble:
    """
    Implements the Pent-View Hybrid-Topology Stacking Ensemble.

    Level 1 Models:
    1. Lexical Bagger (RF): Sparse Text + Metadata
    2. Behavioral Bagger (RF): Sparse History + Metadata
    3. Semantic Booster (XGB): Dense Embeddings + Metadata
    4. Semantic Bagger (RF): Dense Embeddings + Metadata
    5. Contextual Baseline (LR): Metadata Only

    Level 2 Model:
    - Logistic Regression Stacker
    """

    def __init__(self):
        self.models = {
            "lexical_rf": get_lexical_rf(),
            "behavioral_rf": get_behavioral_rf(),
            "semantic_xgb": get_semantic_xgb(),
            "semantic_rf": get_semantic_rf(),
            "contextual_lr": get_contextual_lr(),
        }
        self.meta_learner = get_meta_learner()
        self.n_folds = N_FOLDS
        self.seed = RANDOM_SEED

    def _prepare_view(self, view_name, metadata, specific_data):
        """
        Concatenates global metadata with view-specific data.
        Returns the appropriate matrix format (Sparse CSR or Dense Numpy).
        """
        if view_name == "contextual_lr":
            return metadata

        if view_name in ["lexical_rf", "behavioral_rf"]:
            # Sparse concatenation: hstack sparse matrix with dense metadata
            # Convert metadata to sparse for efficient stacking if needed,
            # but hstack handles mixed types. We convert result to CSR.
            return sparse.hstack([specific_data, metadata]).tocsr()

        if view_name in ["semantic_xgb", "semantic_rf"]:
            # Dense concatenation
            return np.hstack([specific_data, metadata])

        raise ValueError(f"Unknown view name: {view_name}")

    def fit(self, data):
        """
        Trains the ensemble using 5-fold CV stacking.

        Args:
            data (dict): Dictionary containing 'train', 'val', 'test' data splits.
        """
        print_header("Training Stacking Ensemble")

        # Unpack Training Data
        X_meta = data["train"]["metadata"]
        X_lex = data["train"]["lexical"]
        X_beh = data["train"]["behavioral"]
        X_sem = data["train"]["semantic"]
        y = data["train"]["y"]

        # Unpack Validation Data (for XGB early stopping and final evaluation)
        X_val_meta = data["val"]["metadata"]
        X_val_lex = data["val"]["lexical"]
        X_val_beh = data["val"]["behavioral"]
        X_val_sem = data["val"]["semantic"]
        y_val = data["val"]["y"]

        # Initialize OOF Matrix
        n_samples = len(y)
        oof_preds = np.zeros((n_samples, len(self.models)))

        # Cross-Validation
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        print(f"Starting Level 1 Training ({self.n_folds} folds)...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y)):
            with timer(f"Fold {fold + 1}/{self.n_folds}"):
                # Slice Targets
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]

                # Iterate over Level 1 models
                for i, (name, model) in enumerate(self.models.items()):
                    # Prepare Data Views
                    if name == "lexical_rf":
                        X_train_view = self._prepare_view(
                            name, X_meta[train_idx], X_lex[train_idx]
                        )
                        X_val_view = self._prepare_view(
                            name, X_meta[val_idx], X_lex[val_idx]
                        )
                    elif name == "behavioral_rf":
                        X_train_view = self._prepare_view(
                            name, X_meta[train_idx], X_beh[train_idx]
                        )
                        X_val_view = self._prepare_view(
                            name, X_meta[val_idx], X_beh[val_idx]
                        )
                    elif name in ["semantic_xgb", "semantic_rf"]:
                        X_train_view = self._prepare_view(
                            name, X_meta[train_idx], X_sem[train_idx]
                        )
                        X_val_view = self._prepare_view(
                            name, X_meta[val_idx], X_sem[val_idx]
                        )
                    elif name == "contextual_lr":
                        X_train_view = self._prepare_view(name, X_meta[train_idx], None)
                        X_val_view = self._prepare_view(name, X_meta[val_idx], None)

                    # Train
                    if name == "semantic_xgb":
                        model.set_params(
                            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS
                        )
                        model.fit(
                            X_train_view,
                            y_fold_train,
                            eval_set=[(X_val_view, y_fold_val)],
                            verbose=False,
                        )
                    else:
                        model.fit(X_train_view, y_fold_train)

                    # Predict
                    preds = model.predict_proba(X_val_view)[:, 1]
                    oof_preds[val_idx, i] = preds

        # Evaluate Level 1 OOF Performance
        print("\nLevel 1 OOF Performance (AUC):")
        for i, name in enumerate(self.models.keys()):
            auc = roc_auc_score(y, oof_preds[:, i])
            print(f"  {name}: {auc}")

        # Train Level 2 Meta-Learner
        print("\nTraining Level 2 Meta-Learner...")
        self.meta_learner.fit(oof_preds, y)

        # Evaluate Level 2 on OOF
        meta_oof_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        cv_auc = roc_auc_score(y, meta_oof_preds)
        print(f"Level 2 CV AUC: {cv_auc}")

        # ---------------------------------------------------------
        # Retraining on Full Training Set
        # ---------------------------------------------------------
        print("\nRetraining Level 1 Models on Full Training Data...")

        for name, model in self.models.items():
            # Prepare Full Train Views
            if name == "lexical_rf":
                X_full = self._prepare_view(name, X_meta, X_lex)
                model.fit(X_full, y)
            elif name == "behavioral_rf":
                X_full = self._prepare_view(name, X_meta, X_beh)
                model.fit(X_full, y)
            elif name == "semantic_xgb":
                X_full = self._prepare_view(name, X_meta, X_sem)
                # Use Global Validation Set for Early Stopping
                X_val_view = self._prepare_view(name, X_val_meta, X_val_sem)
                model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
                model.fit(
                    X_full,
                    y,
                    eval_set=[(X_val_view, y_val)],
                    verbose=False,
                )
            elif name == "semantic_rf":
                X_full = self._prepare_view(name, X_meta, X_sem)
                model.fit(X_full, y)
            elif name == "contextual_lr":
                X_full = self._prepare_view(name, X_meta, None)
                model.fit(X_full, y)

    def predict(self, data_split):
        """
        Generates predictions for a given data split (val or test).

        Args:
            data_split (dict): Dictionary containing 'metadata', 'lexical', etc.

        Returns:
            np.array: Probability predictions.
        """
        X_meta = data_split["metadata"]
        X_lex = data_split["lexical"]
        X_beh = data_split["behavioral"]
        X_sem = data_split["semantic"]

        n_samples = X_meta.shape[0]
        l1_preds = np.zeros((n_samples, len(self.models)))

        for i, (name, model) in enumerate(self.models.items()):
            if name == "lexical_rf":
                X_view = self._prepare_view(name, X_meta, X_lex)
            elif name == "behavioral_rf":
                X_view = self._prepare_view(name, X_meta, X_beh)
            elif name in ["semantic_xgb", "semantic_rf"]:
                X_view = self._prepare_view(name, X_meta, X_sem)
            elif name == "contextual_lr":
                X_view = self._prepare_view(name, X_meta, None)

            l1_preds[:, i] = model.predict_proba(X_view)[:, 1]

        final_preds = self.meta_learner.predict_proba(l1_preds)[:, 1]
        return final_preds


def generate_submission(data):
    """
    Orchestrates the training, validation, and submission generation.
    """
    ensemble = PentViewStackingEnsemble()

    # Train
    ensemble.fit(data)

    # Validate on Hold-out Set
    print_header("Validation on Hold-out Set")
    val_preds = ensemble.predict(data["val"])
    val_auc = roc_auc_score(data["val"]["y"], val_preds)
    print(f"Hold-out Validation AUC: {val_auc}")

    # Predict on Test
    print_header("Generating Submission")
    test_preds = ensemble.predict(data["test"])

    # Save
    test_ids = data["test"]["ids"]
    submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
