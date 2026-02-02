import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import (
    N_FOLDS,
    SEED,
    SUBMISSION_PATH,
    META_LEARNER_PARAMS,
    ID_COL,
    TARGET_COL,
)
from library.utils import print_info, print_metric, print_header, Timer, save_to_parquet
from library.model_definitions import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
)


class StackingPipeline:
    """
    Manages the training, validation, and inference lifecycle of the Stacking Ensemble.
    Implements the Validation-Guided Retraining Protocol.
    """

    def __init__(self):
        # Initialize Level 1 Base Learners
        self.base_models = [
            LexicalBagger(),
            CommunityBagger(),
            SemanticBooster(),
            SemanticBagger(),
            MetadataAnchor(),
        ]

        # Map models to their specific feature view keys
        self.model_view_map = {
            "LexicalBagger": "lexical",
            "CommunityBagger": "behavioral",
            "SemanticBooster": "semantic",
            "SemanticBagger": "semantic",
            "MetadataAnchor": "metadata",
        }

        # Level 2 Meta-Learner
        self.meta_learner = LogisticRegression(**META_LEARNER_PARAMS)

        # Storage for final retrained models
        self.final_models = []

    def _get_inputs(self, model_name, data_dict):
        """
        Helper to extract the specific view and metadata for a given model.
        """
        view_key = self.model_view_map.get(model_name)
        X_view = data_dict.get(view_key)
        X_meta = data_dict.get("metadata")

        if X_view is None and model_name != "MetadataAnchor":
            print_info(f"Warning: View '{view_key}' not found for {model_name}.")

        return X_view, X_meta

    def run_cross_validation(self, X_train_dict, y_train):
        """
        Performs Stratified K-Fold Cross Validation to generate OOF predictions.
        """
        print_header("Running 5-Fold Cross-Validation (OOF Generation)")

        n_samples = len(y_train)
        n_models = len(self.base_models)

        # Matrix to store OOF predictions: (n_samples, n_models)
        oof_preds = np.zeros((n_samples, n_models))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # We need to index the arrays in the dict, so we ensure they are numpy arrays
        # X_train_dict values are assumed to be (n_samples, n_features)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            print_info(f"--- Fold {fold + 1}/{N_FOLDS} ---")

            y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            for m_idx, base_model in enumerate(self.base_models):
                # Clone model logic is handled by creating new instances or re-fitting
                # Since sklearn models reset on fit, we can reuse the definition class
                # but we need a fresh instance to avoid carrying over state if we were saving them
                # Here we just fit the current instance for the OOF purpose.

                # However, to be clean, we should re-instantiate or clone.
                # Given the custom classes, we will just re-instantiate based on class name for safety
                # or rely on the fact that fit() resets the underlying sklearn/xgb model.
                # The provided classes wrap the model, so calling fit() on the wrapper calls fit() on the underlying model.
                # Sklearn/XGBoost models reset parameters upon fit().

                model_name = base_model.name
                X_view, X_meta = self._get_inputs(model_name, X_train_dict)

                # Slice data for fold
                X_view_tr = X_view[train_idx] if X_view is not None else None
                X_meta_tr = X_meta[train_idx]

                X_view_val = X_view[val_idx] if X_view is not None else None
                X_meta_val = X_meta[val_idx]

                # Fit
                base_model.fit(X_view_tr, X_meta_tr, y_tr_fold)

                # Predict
                probs = base_model.predict_proba(X_view_val, X_meta_val)
                oof_preds[val_idx, m_idx] = probs

        # Evaluate Base Models
        print_header("Base Model CV Performance")
        for m_idx, base_model in enumerate(self.base_models):
            auc = roc_auc_score(y_train, oof_preds[:, m_idx])
            print_metric(f"{base_model.name} OOF AUC", auc)

        return oof_preds

    def train_meta_learner(self, oof_matrix, y_train):
        """
        Trains the Level 2 Logistic Regression on OOF predictions.
        """
        print_header("Training Meta-Learner")

        self.meta_learner.fit(oof_matrix, y_train)

        # Check performance on the training set (OOF)
        meta_preds = self.meta_learner.predict_proba(oof_matrix)[:, 1]
        auc = roc_auc_score(y_train, meta_preds)
        print_metric("Meta-Learner OOF AUC", auc)

        # Print coefficients
        print_info("Meta-Learner Coefficients:")
        for idx, coef in enumerate(self.meta_learner.coef_[0]):
            model_name = self.base_models[idx].name
            print(f"  {model_name}: {coef:.4f}")

    def retrain_final_models(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Implements Validation-Guided Retraining.
        - XGBoost: Train on Train, use Val for Early Stopping.
        - Others: Train on Train + Val.
        """
        print_header("Retraining Final Base Models")

        self.final_models = []  # Reset

        for base_model in self.base_models:
            model_name = base_model.name
            print_info(f"Retraining {model_name}...")

            X_view_train, X_meta_train = self._get_inputs(model_name, X_train_dict)
            X_view_val, X_meta_val = self._get_inputs(model_name, X_val_dict)

            if model_name == "SemanticBooster":
                # XGBoost: Use Validation set for Early Stopping
                # Do NOT concatenate train and val
                base_model.fit(
                    X_view_train,
                    X_meta_train,
                    y_train,
                    X_val_view=X_view_val,
                    X_val_meta=X_meta_val,
                    y_val=y_val,
                )
            else:
                # RF / Linear: Concatenate Train + Val
                if X_view_train is not None and X_view_val is not None:
                    X_view_full = np.vstack([X_view_train, X_view_val])
                else:
                    X_view_full = None

                X_meta_full = np.vstack([X_meta_train, X_meta_val])
                y_full = np.concatenate([y_train, y_val])

                base_model.fit(X_view_full, X_meta_full, y_full)

            self.final_models.append(base_model)

        print_info("All base models retrained successfully.")

    def predict(self, X_test_dict):
        """
        Generates final predictions for the test set.
        1. Generate Level 1 predictions from retrained base models.
        2. Feed to Meta-Learner.
        """
        print_header("Generating Final Predictions")

        n_samples = X_test_dict["metadata"].shape[0]
        n_models = len(self.final_models)

        level1_preds = np.zeros((n_samples, n_models))

        for m_idx, model in enumerate(self.final_models):
            X_view, X_meta = self._get_inputs(model.name, X_test_dict)
            probs = model.predict_proba(X_view, X_meta)
            level1_preds[:, m_idx] = probs

        # Meta-Learner Prediction
        final_probs = self.meta_learner.predict_proba(level1_preds)[:, 1]

        return final_probs

    def generate_submission(self, test_df, predictions):
        """
        Formats and saves the submission file.
        """
        print_header("Saving Submission")

        submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: predictions})

        save_to_parquet(
            submission, SUBMISSION_PATH.replace(".csv", ".parquet")
        )  # Save backup

        # Save as CSV for requirement
        submission.to_csv(SUBMISSION_PATH, index=False)
        print_info(f"Submission saved to {SUBMISSION_PATH}")
        print(submission.head())
