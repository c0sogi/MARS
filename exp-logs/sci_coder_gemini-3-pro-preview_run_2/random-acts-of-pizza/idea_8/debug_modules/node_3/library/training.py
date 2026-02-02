import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, get_logger
from library.data_loader import load_datasets
from library.text_pipeline import ProjectedTextEmbedder
from library.tabular_pipeline import InteractionMetadataProcessor
from library.model import create_bagged_ensemble


class Trainer:
    """
    Manages the training, validation, and submission generation process
    for the Projected Semantic-Interaction Ensemble.
    """

    def __init__(self, load_cached_data=True):
        self.logger = get_logger("Trainer")
        self.load_cached_data = load_cached_data
        self.text_pipeline = ProjectedTextEmbedder()
        self.meta_pipeline = InteractionMetadataProcessor()
        set_seed(Config.RANDOM_SEED)

    def _prepare_data(self, use_all_data=False):
        """
        Prepares features for training.

        Args:
            use_all_data (bool): If True, combines train and validation sets for training.

        Returns:
            tuple: (X_train, y_train, X_eval, y_eval, X_test, test_ids)
                   Note: if use_all_data is True, X_eval/y_eval will be None.
        """
        # Load raw dataframes
        df_train, df_val, df_test = load_datasets(
            load_cached_data=self.load_cached_data
        )

        if use_all_data:
            self.logger.info("Combining Train and Validation sets for full training.")
            df_train_full = pd.concat([df_train, df_val], ignore_index=True)

            # 1. Text Pipeline
            # Fit on full data
            self.text_pipeline.fit(
                df_train_full, load_cached_data=self.load_cached_data
            )
            # Transform full data (using specific split name to manage cache correctly)
            X_text_train = self.text_pipeline.transform(
                df_train_full,
                split="train_full",
                load_cached_data=self.load_cached_data,
            )

            # 2. Metadata Pipeline
            # Fit on full data
            self.meta_pipeline.fit(
                df_train_full, load_cached_data=self.load_cached_data
            )
            # Transform full data
            X_meta_train = self.meta_pipeline.transform(
                df_train_full,
                split="train_full",
                load_cached_data=self.load_cached_data,
            )

            y_train = df_train_full["requester_received_pizza"].values

            # 3. Test Data
            X_text_test = self.text_pipeline.transform(
                df_test, split="test", load_cached_data=self.load_cached_data
            )
            X_meta_test = self.meta_pipeline.transform(
                df_test, split="test", load_cached_data=self.load_cached_data
            )

            # 4. Fusion
            X_train_combined = np.hstack([X_text_train, X_meta_train])
            X_test_combined = np.hstack([X_text_test, X_meta_test])

            return (
                X_train_combined,
                y_train,
                None,
                None,
                X_test_combined,
                df_test["request_id"],
            )

        else:
            self.logger.info("Using standard Train/Val split.")

            # 1. Text Pipeline
            # Fit on standard train
            self.text_pipeline.fit(df_train, load_cached_data=self.load_cached_data)
            X_text_train = self.text_pipeline.transform(
                df_train, split="train", load_cached_data=self.load_cached_data
            )
            X_text_val = self.text_pipeline.transform(
                df_val, split="val", load_cached_data=self.load_cached_data
            )

            # 2. Metadata Pipeline
            # Fit on standard train
            self.meta_pipeline.fit(df_train, load_cached_data=self.load_cached_data)
            X_meta_train = self.meta_pipeline.transform(
                df_train, split="train", load_cached_data=self.load_cached_data
            )
            X_meta_val = self.meta_pipeline.transform(
                df_val, split="val", load_cached_data=self.load_cached_data
            )

            y_train = df_train["requester_received_pizza"].values
            y_val = df_val["requester_received_pizza"].values

            # 3. Test Data
            X_text_test = self.text_pipeline.transform(
                df_test, split="test", load_cached_data=self.load_cached_data
            )
            X_meta_test = self.meta_pipeline.transform(
                df_test, split="test", load_cached_data=self.load_cached_data
            )

            # 4. Fusion
            X_train_combined = np.hstack([X_text_train, X_meta_train])
            X_val_combined = np.hstack([X_text_val, X_meta_val])
            X_test_combined = np.hstack([X_text_test, X_meta_test])

            return (
                X_train_combined,
                y_train,
                X_val_combined,
                y_val,
                X_test_combined,
                df_test["request_id"],
            )

    def cross_validate(self, n_splits=5):
        """
        Performs Stratified K-Fold Cross-Validation on the training set.
        """
        X_train, y_train, _, _, _, _ = self._prepare_data(use_all_data=False)

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=Config.RANDOM_SEED
        )
        fold_aucs = []

        self.logger.info(f"Starting {n_splits}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model = create_bagged_ensemble()
            model.fit(X_tr, y_tr)

            preds = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, preds)
            fold_aucs.append(auc)

            self.logger.info(f"Fold {fold+1} ROC AUC: {auc:.15f}")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        self.logger.info(f"Mean CV ROC AUC: {mean_auc:.15f} (+/- {std_auc:.15f})")
        return mean_auc

    def train_final(self, use_all_data=True):
        """
        Trains the final model and generates the submission file.

        Args:
            use_all_data (bool): If True, trains on combined train+val sets.
                                 If False, trains on train set and evaluates on val set.
        """
        X_train, y_train, X_val, y_val, X_test, request_ids = self._prepare_data(
            use_all_data=use_all_data
        )

        self.logger.info("Training final model...")
        model = create_bagged_ensemble()
        model.fit(X_train, y_train)

        if not use_all_data and X_val is not None:
            self.logger.info("Evaluating on holdout validation set...")
            val_preds = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            self.logger.info(f"Holdout Validation ROC AUC: {val_auc:.15f}")

        self.logger.info("Generating predictions for test set...")
        test_probs = model.predict_proba(X_test)[:, 1]

        submission = pd.DataFrame(
            {"request_id": request_ids, "requester_received_pizza": test_probs}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
