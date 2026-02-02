import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import set_seed
from library.model_factory import build_bagged_logistic_regression, build_meta_learner


class StackingEngine:
    """
    Orchestrates the Multi-View Stacked Generalization Strategy.
    Manages Stratified K-Fold CV, hyperparameter tuning for base experts,
    meta-learner training, and submission generation.
    """

    def __init__(self, n_folds=Config.N_FOLDS, random_state=Config.RANDOM_SEED):
        self.n_folds = n_folds
        self.random_state = random_state
        self.text_grid = Config.TEXT_EXPERT_GRID
        self.meta_grid = Config.META_EXPERT_GRID
        set_seed(self.random_state)

    def _get_best_estimator(self, X, y, param_grid, base_builder_func):
        """
        Performs GridSearchCV to find the best hyperparameters for a view.

        Args:
            X: Feature matrix.
            y: Target vector.
            param_grid: Dictionary of parameters to tune.
            base_builder_func: Function to build the base estimator.

        Returns:
            The best fitted estimator.
        """
        # Create a base model instance to wrap in GridSearchCV
        # We use default params initially; GridSearch will override relevant ones
        base_model = base_builder_func(random_state=self.random_state)

        # Adjust param_grid keys to match the nested structure of BaggingClassifier
        # BaggingClassifier wraps the base estimator in 'estimator' (sklearn >= 1.2)
        # or 'base_estimator' (older). model_factory uses 'estimator'.
        # We need to map e.g., 'C' to 'estimator__C'.

        # However, BaggingClassifier params (like n_estimators) are top-level.
        # The Config grids mix these. We assume the grid provided in Config
        # targets the inner LogisticRegression for 'C', 'penalty', 'solver', 'class_weight'.

        sklearn_grid = {}
        for key, values in param_grid.items():
            if key in ["C", "penalty", "solver", "class_weight"]:
                sklearn_grid[f"estimator__{key}"] = values
            else:
                sklearn_grid[key] = values

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=sklearn_grid,
            scoring="roc_auc",
            cv=3,  # Inner CV for tuning
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def run(
        self,
        X_text_train,
        X_meta_train,
        y_train,
        X_text_val,
        X_meta_val,
        y_val,
        X_text_test,
        X_meta_test,
    ):
        """
        Executes the stacking pipeline.

        Args:
            X_text_train, X_meta_train: Training features for View A and B.
            y_train: Training labels.
            X_text_val, X_meta_val: Validation features.
            y_val: Validation labels.
            X_text_test, X_meta_test: Test features.
        """
        set_seed(self.random_state)

        # Ensure inputs are numpy arrays
        X_meta_train = (
            X_meta_train.values if hasattr(X_meta_train, "values") else X_meta_train
        )
        X_meta_val = X_meta_val.values if hasattr(X_meta_val, "values") else X_meta_val
        X_meta_test = (
            X_meta_test.values if hasattr(X_meta_test, "values") else X_meta_test
        )

        y_train = np.array(y_train)
        y_val = np.array(y_val)

        # Initialize containers for OOF and Test predictions
        # Columns: 0 -> Text Expert, 1 -> Metadata Expert
        oof_preds = np.zeros((len(y_train), 2))

        # We accumulate test/val predictions from each fold to average later
        val_preds_accum = np.zeros((len(y_val), 2))
        test_preds_accum = np.zeros((X_text_test.shape[0], 2))

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        print(f"Starting Stacking with {self.n_folds} folds...")

        for fold, (train_idx, holdout_idx) in enumerate(
            skf.split(X_text_train, y_train)
        ):
            print(f"Processing Fold {fold + 1}/{self.n_folds}")

            # Split Data
            X_text_tr_fold, X_text_ho_fold = (
                X_text_train[train_idx],
                X_text_train[holdout_idx],
            )
            X_meta_tr_fold, X_meta_ho_fold = (
                X_meta_train[train_idx],
                X_meta_train[holdout_idx],
            )
            y_tr_fold, y_ho_fold = y_train[train_idx], y_train[holdout_idx]

            # --- View A: Text Expert ---
            # Tune and Train
            best_text_model = self._get_best_estimator(
                X_text_tr_fold,
                y_tr_fold,
                self.text_grid,
                build_bagged_logistic_regression,
            )

            # Predict OOF (Holdout)
            oof_preds[holdout_idx, 0] = best_text_model.predict_proba(X_text_ho_fold)[
                :, 1
            ]

            # Predict Val and Test (Accumulate)
            val_preds_accum[:, 0] += best_text_model.predict_proba(X_text_val)[:, 1]
            test_preds_accum[:, 0] += best_text_model.predict_proba(X_text_test)[:, 1]

            # --- View B: Metadata Expert ---
            # Tune and Train
            best_meta_model = self._get_best_estimator(
                X_meta_tr_fold,
                y_tr_fold,
                self.meta_grid,
                build_bagged_logistic_regression,
            )

            # Predict OOF (Holdout)
            oof_preds[holdout_idx, 1] = best_meta_model.predict_proba(X_meta_ho_fold)[
                :, 1
            ]

            # Predict Val and Test (Accumulate)
            val_preds_accum[:, 1] += best_meta_model.predict_proba(X_meta_val)[:, 1]
            test_preds_accum[:, 1] += best_meta_model.predict_proba(X_meta_test)[:, 1]

        # Average predictions for Val and Test
        val_preds_avg = val_preds_accum / self.n_folds
        test_preds_avg = test_preds_accum / self.n_folds

        print("Base Learners Training Complete.")

        # --- Meta Learner Training ---
        print("Training Meta-Learner...")

        # We use the OOF predictions as training data for the meta-learner
        meta_learner = build_meta_learner(**Config.STACKING_META_PARAMS)
        meta_learner.fit(oof_preds, y_train)

        # Evaluate on Training OOF (Consistency Check)
        train_auc = roc_auc_score(y_train, meta_learner.predict_proba(oof_preds)[:, 1])
        print(f"Meta-Learner OOF AUC: {train_auc}")

        # Evaluate on Validation Set
        # We use the averaged predictions from base learners as input
        final_val_probs = meta_learner.predict_proba(val_preds_avg)[:, 1]
        val_auc = roc_auc_score(y_val, final_val_probs)
        print(f"Meta-Learner Validation AUC: {val_auc}")

        # --- Submission Generation ---
        print("Generating Submission...")
        final_test_probs = meta_learner.predict_proba(test_preds_avg)[:, 1]

        self._save_submission(final_test_probs)

    def _save_submission(self, predictions):
        """
        Saves the predictions to the submission file format.
        """
        # Load test metadata to get request_ids
        df_test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Ensure alignment
        if len(df_test_meta) != len(predictions):
            raise ValueError(
                f"Mismatch in prediction length: Metadata {len(df_test_meta)} vs Preds {len(predictions)}"
            )

        submission = pd.DataFrame(
            {
                "request_id": df_test_meta["request_id"],
                "requester_received_pizza": predictions,
            }
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
