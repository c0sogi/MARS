import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from library.config import (
    RF_LEXICAL_PARAMS,
    RF_BEHAVIORAL_PARAMS,
    XGB_SEMANTIC_PARAMS,
    XGB_EARLY_STOPPING_ROUNDS,
    RF_SEMANTIC_PARAMS,
    LR_ANCHOR_PARAMS,
    META_LEARNER_PARAMS,
    SEED,
    N_FOLDS,
    N_JOBS,
    SUBMISSION_DIR,
)
from library.utils import get_logger

logger = get_logger("models")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_features(data_dict, split, views):
    """
    Retrieves and concatenates specified feature views for a given split.
    Handles 'full' split by vertically stacking 'train' and 'val'.
    Handles mixing sparse and dense features by horizontally stacking.
    """

    def load_view(s, v):
        key = f"X_{s}_{v}"
        if key not in data_dict:
            raise KeyError(f"Key {key} not found in data dictionary.")
        return data_dict[key]

    # Determine splits to load
    if split == "full":
        splits_to_load = ["train", "val"]
    else:
        splits_to_load = [split]

    # Load and vertically stack rows (if full) for each view
    view_data_list = []
    for view in views:
        parts = [load_view(s, view) for s in splits_to_load]

        # Check if sparse or dense
        if sparse.issparse(parts[0]):
            combined_view = sparse.vstack(parts)
        else:
            combined_view = np.vstack(parts)
        view_data_list.append(combined_view)

    # Horizontally stack views
    # If any view is sparse, result is sparse
    is_any_sparse = any(sparse.issparse(v) for v in view_data_list)

    if is_any_sparse:
        # Convert dense to sparse csc/csr before stacking if mixed
        final_parts = []
        for v in view_data_list:
            if not sparse.issparse(v):
                final_parts.append(sparse.csr_matrix(v))
            else:
                final_parts.append(v)
        X_combined = sparse.hstack(final_parts)
    else:
        X_combined = np.hstack(view_data_list)

    return X_combined


def _get_targets(data_dict, split):
    """Retrieves targets for a split (or combined full split)."""
    if split == "full":
        return np.concatenate([data_dict["y_train"], data_dict["y_val"]])
    else:
        return data_dict[f"y_{split}"]


# =============================================================================
# BASE LEARNERS
# =============================================================================


class LexicalBagger(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier(**RF_LEXICAL_PARAMS)
        self.views = ["lexical", "metadata"]

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CommunityBagger(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier(**RF_BEHAVIORAL_PARAMS)
        self.views = ["community", "metadata"]

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SemanticBooster(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model_params = XGB_SEMANTIC_PARAMS.copy()
        self.model = None
        self.views = ["semantic", "metadata"]

    def fit(self, X, y, eval_set=None):
        # Calculate dynamic scale_pos_weight
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        params = self.model_params.copy()
        params["scale_pos_weight"] = scale_pos_weight

        self.model = XGBClassifier(**params)

        if eval_set:
            self.model.fit(
                X,
                y,
                eval_set=eval_set,
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SemanticBagger(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier(**RF_SEMANTIC_PARAMS)
        self.views = ["semantic", "metadata"]

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MetadataAnchor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = LogisticRegression(**LR_ANCHOR_PARAMS)
        self.views = ["metadata"]

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================


class StackingEnsemble:
    def __init__(self):
        self.base_learners = {
            "lexical_bagger": LexicalBagger(),
            "community_bagger": CommunityBagger(),
            "semantic_booster": SemanticBooster(),
            "semantic_bagger": SemanticBagger(),
            "metadata_anchor": MetadataAnchor(),
        }
        self.meta_learner = LogisticRegression(**META_LEARNER_PARAMS)
        self.trained_base_learners = {}

    def fit(self, data_dict):
        logger.info("Starting Stacking Ensemble Training...")

        # 1. OOF Generation (Level 1)
        # We use the 'train' split from metadata for CV
        X_train_indices = np.arange(len(data_dict["y_train"]))
        y_train_full = data_dict["y_train"]

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # Placeholder for OOF predictions: (n_samples, n_models)
        oof_preds = pd.DataFrame(
            0.0, index=np.arange(len(y_train_full)), columns=self.base_learners.keys()
        )

        logger.info(f"Generating OOF predictions using {N_FOLDS}-Fold CV...")

        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X_train_indices, y_train_full)
        ):
            y_tr, y_va = y_train_full[train_idx], y_train_full[val_idx]

            fold_preds = {}

            for name, learner in self.base_learners.items():
                # Prepare data for this fold
                # We need to slice the raw features based on indices
                # Since features are pre-loaded in data_dict as full arrays, we slice them.

                # Helper to slice a specific view
                def get_fold_data(indices, views):
                    # We reconstruct the feature matrix for the full train set first
                    # This is slightly inefficient but safe.
                    # Optimization: slice directly from data_dict['X_train_view']

                    # Construct the full X for this view for the 'train' split
                    X_full_view = _get_features(data_dict, "train", views)

                    # Slice
                    if sparse.issparse(X_full_view):
                        return X_full_view[indices]
                    else:
                        return X_full_view[indices]

                X_tr = get_fold_data(train_idx, learner.views)
                X_va = get_fold_data(val_idx, learner.views)

                # Clone and Fit
                # Note: We don't use early stopping for XGB in the CV loop to keep it simple/standard
                # or we could split train_idx further. Here we fit standardly.
                model = clone(learner)

                # For SemanticBooster in CV, we can use a portion of X_tr as eval or just fit
                # To match protocol, we just fit. XGB params control overfitting.
                model.fit(X_tr, y_tr)

                # Predict
                p = model.predict_proba(X_va)[:, 1]
                oof_preds.loc[val_idx, name] = p
                fold_preds[name] = p

            # Calculate Fold AUC for Meta (approx)
            # Simple average of base learners for logging
            avg_p = np.mean(list(fold_preds.values()), axis=0)
            fold_auc = roc_auc_score(y_va, avg_p)
            fold_aucs.append(fold_auc)
            logger.info(f"Fold {fold+1}/{N_FOLDS} - Avg Base AUC: {fold_auc:.6f}")

        logger.info(f"Mean OOF AUC (Avg Base): {np.mean(fold_aucs):.6f}")

        # 2. Train Meta-Learner (Level 2)
        logger.info("Training Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y_train_full)

        meta_auc = roc_auc_score(
            y_train_full, self.meta_learner.predict_proba(oof_preds)[:, 1]
        )
        logger.info(f"Meta-Learner OOF AUC: {meta_auc:.10f}")

        logger.info("Meta-Learner Coefficients:")
        for name, coef in zip(self.base_learners.keys(), self.meta_learner.coef_[0]):
            logger.info(f"  {name}: {coef:.4f}")

        # 3. Retrain Base Learners (Final Retraining)
        logger.info("Retraining Base Learners on Full Data...")

        for name, learner in self.base_learners.items():
            logger.info(f"Retraining {name}...")
            final_model = clone(learner)

            if name == "semantic_booster":
                # XGBoost: Train on 'train', Eval on 'val' (Global splits)
                X_tr = _get_features(data_dict, "train", learner.views)
                y_tr = data_dict["y_train"]
                X_va = _get_features(data_dict, "val", learner.views)
                y_va = data_dict["y_val"]

                final_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            else:
                # Others: Train on Full (Train + Val)
                X_full = _get_features(data_dict, "full", learner.views)
                y_full = _get_targets(data_dict, "full")

                final_model.fit(X_full, y_full)

            self.trained_base_learners[name] = final_model

        logger.info("Training Complete.")

    def predict_proba(self, data_dict, split="test"):
        """
        Generates probability predictions for a specific split.
        """
        # 1. Generate Base Predictions
        base_preds = pd.DataFrame(
            index=np.arange(len(_get_features(data_dict, split, ["metadata"])))
        )  # Dummy to get length

        for name, model in self.trained_base_learners.items():
            # Get features for this split
            # We need to access the views required by this specific model
            # We can't access model.views directly if model is a sklearn wrapper,
            # but our wrapper classes store it.

            # Since we cloned, the attribute should be there.
            views = model.views
            X = _get_features(data_dict, split, views)
            base_preds[name] = model.predict_proba(X)[:, 1]

        # 2. Meta Prediction
        final_probs = self.meta_learner.predict_proba(base_preds)[:, 1]
        return final_probs

    def generate_submission(self, data_dict):
        """
        Generates predictions for test set and saves to submission file.
        """
        logger.info("Generating submission...")

        # Get Test IDs
        test_ids = data_dict["id_test"]

        # Predict
        probs = self.predict_proba(data_dict, split="test")

        # Create DataFrame
        sub_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": probs}
        )

        # Save
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        logger.info(f"Submission saved to {save_path}")

        # Verify
        logger.info(f"Submission Head:\n{sub_df.head()}")
