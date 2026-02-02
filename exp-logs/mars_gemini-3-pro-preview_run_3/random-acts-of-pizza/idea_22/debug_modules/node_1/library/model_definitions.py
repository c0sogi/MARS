import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from library.config import Config
from library.utils import set_seed


class HexEnsemble:
    """
    Hex-View Hybrid-Topology Stacking Ensemble.

    Architecture:
    - Level 1: 6 Base Learners processing specific feature views (Lexical, Behavioral, Semantic, Manifold, Metadata).
    - Level 2: Logistic Regression Meta-Learner stacking Level 1 probabilities.

    Implements Validation-Guided Retraining to prevent overfitting.
    """

    def __init__(self):
        set_seed(Config.SEED)
        self.base_learners = {}
        self.meta_learner = LogisticRegression(**Config.LR_PARAMS)
        self._initialize_models()

    def _initialize_models(self):
        """Initializes the 6 base learners with configurations from Config."""
        # 1. Sparse Lexical Branch (Text TF-IDF + Metadata)
        self.base_learners["lexical_bagger"] = RandomForestClassifier(
            **Config.RF_PARAMS
        )

        # 2. Sparse Behavioral Branch (History TF-IDF + Metadata)
        self.base_learners["community_bagger"] = RandomForestClassifier(
            **Config.RF_PARAMS
        )

        # 3. Dense Semantic Branch (XGBoost on Embeddings + Metadata)
        self.base_learners["semantic_booster"] = XGBClassifier(**Config.XGB_PARAMS)

        # 4. Dense Semantic Branch (Random Forest on Embeddings + Metadata)
        self.base_learners["semantic_bagger"] = RandomForestClassifier(
            **Config.RF_PARAMS
        )

        # 5. Manifold Branch (kNN on PCA Embeddings + Metadata)
        self.base_learners["manifold_neighbor"] = KNeighborsClassifier(
            **Config.KNN_PARAMS
        )

        # 6. Contextual Branch (Logistic Regression on Metadata)
        self.base_learners["metadata_anchor"] = LogisticRegression(**Config.LR_PARAMS)

    def _construct_input(self, model_name, data_view):
        """
        Constructs the specific feature matrix X for a given base learner.
        Concatenates the modality-specific view with the global metadata vector.

        Args:
            model_name (str): Name of the base learner.
            data_view (dict): Dictionary containing feature arrays ('lexical', 'metadata', etc.).

        Returns:
            np.ndarray or scipy.sparse.csr_matrix: The concatenated feature matrix.
        """
        meta = data_view["metadata"]

        if model_name == "lexical_bagger":
            # Sparse Lexical + Dense Metadata -> Sparse
            return scipy.sparse.hstack([data_view["lexical"], meta]).tocsr()

        elif model_name == "community_bagger":
            # Sparse Behavioral + Dense Metadata -> Sparse
            return scipy.sparse.hstack([data_view["behavioral"], meta]).tocsr()

        elif model_name in ["semantic_booster", "semantic_bagger"]:
            # Dense Semantic + Dense Metadata -> Dense
            return np.hstack([data_view["semantic"], meta])

        elif model_name == "manifold_neighbor":
            # Dense Manifold (PCA) + Dense Metadata -> Dense
            return np.hstack([data_view["manifold"], meta])

        elif model_name == "metadata_anchor":
            # Just Metadata -> Dense
            return meta

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def fit(self, data_train, data_val):
        """
        Executes the Validation-Guided Retraining Protocol.

        1. OOF Generation: 5-Fold CV on data_train to generate inputs for Meta-Learner.
        2. Meta-Learner Training: Train L2 model on OOF predictions.
        3. Final Retraining: Retrain L1 models on Full Data (Train + Val).
           - XGBoost uses the explicit Val set for Early Stopping.

        Args:
            data_train (dict): Training data dictionary from FeaturePipeline.
            data_val (dict): Validation data dictionary from FeaturePipeline.
        """
        print("Starting HexEnsemble Training...")

        y_train = data_train["y"]
        n_samples = len(y_train)
        model_names = list(self.base_learners.keys())
        n_models = len(model_names)

        # ---------------------------------------------------------------------
        # Phase 1: OOF Generation (Level 1 Training)
        # ---------------------------------------------------------------------
        print(f"Phase 1: Generating OOF predictions with {Config.N_FOLDS}-Fold CV...")
        oof_preds = np.zeros((n_samples, n_models))
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Helper to slice data dictionaries
        def slice_view(view, indices):
            sliced = {}
            for k, v in view.items():
                if k == "y":
                    continue
                if scipy.sparse.issparse(v):
                    sliced[k] = v[indices]
                else:
                    sliced[k] = v[indices]
            return sliced

        for fold, (train_idx, valid_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            y_fold_train = y_train[train_idx]
            y_fold_valid = y_train[valid_idx]

            train_view = slice_view(data_train, train_idx)
            valid_view = slice_view(data_train, valid_idx)

            for col_idx, name in enumerate(model_names):
                model = clone(self.base_learners[name])

                X_fold_train = self._construct_input(name, train_view)
                X_fold_valid = self._construct_input(name, valid_view)

                # Dynamic scale_pos_weight for XGBoost
                if name == "semantic_booster":
                    n_pos = np.sum(y_fold_train)
                    n_neg = len(y_fold_train) - n_pos
                    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
                    model.set_params(scale_pos_weight=scale_weight)

                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_valid, y_fold_valid)],
                        early_stopping_rounds=Config.XGB_EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    )
                else:
                    model.fit(X_fold_train, y_fold_train)

                # Predict
                preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_preds[valid_idx, col_idx] = preds

        # Print OOF Metrics
        print("\n--- Level 1 OOF Performance (AUC) ---")
        for i, name in enumerate(model_names):
            auc = roc_auc_score(y_train, oof_preds[:, i])
            print(f"{name}: {auc:.16f}")

        # ---------------------------------------------------------------------
        # Phase 2: Level 2 Training (Meta Learner)
        # ---------------------------------------------------------------------
        print("\nPhase 2: Training Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y_train)

        meta_oof_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(y_train, meta_oof_preds)
        print(f"Meta-Learner OOF AUC: {meta_auc:.16f}")

        # ---------------------------------------------------------------------
        # Phase 3: Final Retraining on Full Data (Train + Val)
        # ---------------------------------------------------------------------
        print("\nPhase 3: Retraining Base Learners on Full Data (Train + Val)...")

        # Combine Train and Val for final training
        y_full = np.concatenate([data_train["y"], data_val["y"]])

        def concat_views(v1, v2):
            combined = {}
            for k in v1.keys():
                if k == "y":
                    continue
                if scipy.sparse.issparse(v1[k]):
                    combined[k] = scipy.sparse.vstack([v1[k], v2[k]])
                else:
                    combined[k] = np.concatenate([v1[k], v2[k]])
            return combined

        data_full = concat_views(data_train, data_val)

        for name in model_names:
            X_full = self._construct_input(name, data_full)

            if name == "semantic_booster":
                # For XGBoost, we use the Validation set (which is part of X_full)
                # as the eval_set to trigger early stopping.
                X_val_only = self._construct_input(name, data_val)
                y_val_only = data_val["y"]

                # Recalculate scale_pos_weight for full dataset
                n_pos = np.sum(y_full)
                n_neg = len(y_full) - n_pos
                scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
                self.base_learners[name].set_params(scale_pos_weight=scale_weight)

                self.base_learners[name].fit(
                    X_full,
                    y_full,
                    eval_set=[(X_val_only, y_val_only)],
                    early_stopping_rounds=Config.XGB_EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            else:
                self.base_learners[name].fit(X_full, y_full)

        print("HexEnsemble Training Complete.")

    def predict(self, data_test):
        """
        Generates final predictions for the test set using the stacked ensemble.

        Args:
            data_test (dict): Test data dictionary from FeaturePipeline.

        Returns:
            np.ndarray: Probability of success (Class 1).
        """
        n_samples = data_test["metadata"].shape[0]
        model_names = list(self.base_learners.keys())
        n_models = len(model_names)

        L1_preds = np.zeros((n_samples, n_models))

        for col_idx, name in enumerate(model_names):
            X_test = self._construct_input(name, data_test)
            L1_preds[:, col_idx] = self.base_learners[name].predict_proba(X_test)[:, 1]

        final_preds = self.meta_learner.predict_proba(L1_preds)[:, 1]
        return final_preds

    def generate_submission(self, data_test, test_ids):
        """
        Generates predictions and saves them to the submission file.

        Args:
            data_test (dict): Test data dictionary.
            test_ids (array-like): List of request_ids corresponding to data_test.
        """
        print("Generating submission file...")
        preds = self.predict(data_test)

        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: preds}
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
