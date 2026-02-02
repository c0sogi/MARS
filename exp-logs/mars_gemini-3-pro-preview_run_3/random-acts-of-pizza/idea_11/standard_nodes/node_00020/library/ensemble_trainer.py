import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.model_definitions import (
    SparseRandomForest,
    DenseXGBoost,
    StackingMetaLearner,
)


class StackingEnsemble:
    """
    Orchestrates the Tri-View Topology-Matched Stacking Ensemble.

    Responsibilities:
    1. Combine Train and Validation sets for maximum data utilization.
    2. Perform 5-Fold Cross-Validation to generate Out-Of-Fold (OOF) predictions.
    3. Train Level 1 Base Learners (Lexical RF, Behavioral RF, Contextual XGB).
    4. Train Level 2 Meta-Learner (Logistic Regression) on OOF predictions.
    5. Retrain all Level 1 models on the full dataset.
    6. Generate predictions for the Test set and save submission.
    """

    def __init__(self):
        # Level 1 Models
        self.lexical_model = SparseRandomForest()
        self.behavioral_model = SparseRandomForest()
        self.contextual_model = DenseXGBoost()

        # Level 2 Meta-Learner
        self.meta_learner = StackingMetaLearner()

        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_SEED

    def _concat_data(self, data_dict):
        """
        Concatenates Train and Validation data to form a single training set for CV.
        """
        print("Concatenating Train and Validation sets for Cross-Validation...")

        # Lexical (Sparse)
        X_lexical = sparse.vstack(
            [data_dict["X_train_lexical"], data_dict["X_val_lexical"]]
        ).tocsr()

        # Behavioral (Sparse)
        X_behavioral = sparse.vstack(
            [data_dict["X_train_behavioral"], data_dict["X_val_behavioral"]]
        ).tocsr()

        # Dense (Numpy)
        X_dense = np.concatenate(
            [data_dict["X_train_dense"], data_dict["X_val_dense"]], axis=0
        )

        # Targets
        y = np.concatenate([data_dict["y_train"], data_dict["y_val"]], axis=0)

        return X_lexical, X_behavioral, X_dense, y

    def fit(self, data_dict):
        """
        Executes the stacking training pipeline.

        Args:
            data_dict (dict): Dictionary containing features and targets for train/val/test.
        """
        # 1. Prepare Data
        X_lexical, X_behavioral, X_dense, y = self._concat_data(data_dict)

        n_samples = y.shape[0]
        # OOF Matrix: [n_samples, 3 models]
        oof_preds = np.zeros((n_samples, 3))

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        print(f"\nStarting {self.n_folds}-Fold Cross-Validation Stacking...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dense, y)):
            print(f"  Processing Fold {fold + 1}/{self.n_folds}...")

            # --- Slice Data for Fold ---
            # Lexical
            X_lex_train = X_lexical[train_idx]
            X_lex_val = X_lexical[val_idx]

            # Behavioral
            X_beh_train = X_behavioral[train_idx]
            X_beh_val = X_behavioral[val_idx]

            # Dense
            X_dense_train = X_dense[train_idx]
            X_dense_val = X_dense[val_idx]

            # Targets
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # --- Train Level 1 Models ---

            # 1. Lexical Bagger (RF)
            self.lexical_model.fit(X_lex_train, y_train_fold)
            p_lex = self.lexical_model.predict_proba(X_lex_val)
            oof_preds[val_idx, 0] = p_lex

            # 2. Behavioral Bagger (RF)
            self.behavioral_model.fit(X_beh_train, y_train_fold)
            p_beh = self.behavioral_model.predict_proba(X_beh_val)
            oof_preds[val_idx, 1] = p_beh

            # 3. Contextual Booster (XGB)
            # Use fold validation set for early stopping
            self.contextual_model.fit(
                X_dense_train, y_train_fold, eval_set=[(X_dense_val, y_val_fold)]
            )
            p_ctx = self.contextual_model.predict_proba(X_dense_val)
            oof_preds[val_idx, 2] = p_ctx

        # --- Evaluate OOF Performance ---
        overall_auc = roc_auc_score(
            y, oof_preds.mean(axis=1)
        )  # Simple average for quick check
        print(f"\nOOF Score (Simple Average): AUC = {overall_auc}")

        # --- Train Level 2 Meta-Learner ---
        print("Training Level 2 Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y)

        meta_preds = self.meta_learner.predict_proba(oof_preds)
        meta_auc = roc_auc_score(y, meta_preds)
        print(f"OOF Score (Stacked Meta-Learner): AUC = {meta_auc}")

        # --- Retrain Level 1 Models on Full Data ---
        print("\nRetraining Level 1 Base Learners on full dataset...")

        # Note: We fit on the combined Train+Val set.
        # For XGBoost, we don't have a separate validation set here, so we fit without early stopping
        # relying on the hyperparameters (n_estimators) and ensemble robustness.

        self.lexical_model.fit(X_lexical, y)
        self.behavioral_model.fit(X_behavioral, y)
        self.contextual_model.fit(X_dense, y, eval_set=None)

        print("Training complete.")

    def predict(self, data_dict):
        """
        Generates predictions for the test set.

        Args:
            data_dict (dict): Dictionary containing test features.

        Returns:
            np.ndarray: Final probability predictions for the test set.
        """
        print("\nGenerating predictions for Test set...")

        X_lex_test = data_dict["X_test_lexical"]
        X_beh_test = data_dict["X_test_behavioral"]
        X_dense_test = data_dict["X_test_dense"]

        n_test = X_dense_test.shape[0]
        L1_preds = np.zeros((n_test, 3))

        # 1. Lexical
        L1_preds[:, 0] = self.lexical_model.predict_proba(X_lex_test)

        # 2. Behavioral
        L1_preds[:, 1] = self.behavioral_model.predict_proba(X_beh_test)

        # 3. Contextual
        L1_preds[:, 2] = self.contextual_model.predict_proba(X_dense_test)

        # Level 2
        final_preds = self.meta_learner.predict_proba(L1_preds)

        return final_preds

    def save_submission(self, predictions):
        """
        Saves the predictions to a CSV file in the required format.

        Args:
            predictions (np.ndarray): Predicted probabilities.
        """
        # Load test metadata to get request_ids
        # We need to ensure the order matches the features.
        # FeatureManager processes test.parquet sequentially, so order is preserved.
        test_df = pd.read_parquet(Config.TEST_PATH)

        if Config.DEBUG:
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        if len(test_df) != len(predictions):
            raise ValueError(
                f"Shape mismatch: Test DF has {len(test_df)} rows, predictions has {len(predictions)}."
            )

        submission_df = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: predictions}
        )

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
