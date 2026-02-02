import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import DataLoader
from library.embedder import EmbeddingGenerator
from library.feature_processor import WhitenedFusionPipeline
from library.model_builder import ModelBuilder


class CrossValidationTrainer:
    """
    Orchestrates the 5-fold Stratified Cross-Validation training and inference loop
    for the Whitened Multi-Field Asymmetric Dual-Backbone Ensemble.
    """

    def __init__(self):
        self.logger = setup_logger("CrossValidationTrainer")
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED
        set_seed(self.seed)

    def run(self, load_cached_data: bool = True):
        """
        Executes the full training and evaluation pipeline.

        Args:
            load_cached_data (bool): Whether to use cached data/embeddings.
        """
        self.logger.info("Starting Cross-Validation Training Run...")

        # 1. Load Data
        # We merge the metadata-defined train and val splits to perform our own 5-fold CV
        loader = DataLoader()
        train_split_df, val_split_df, test_df = loader.load_data(
            load_cached=load_cached_data
        )

        # 2. Generate/Load Embeddings
        embedder = EmbeddingGenerator()
        embeddings = embedder.generate_embeddings(
            train_split_df, val_split_df, test_df, load_cached=load_cached_data
        )

        # 3. Consolidate Development Data (Train + Val) for CV
        # Concatenate DataFrames
        dev_df = pd.concat([train_split_df, val_split_df], axis=0).reset_index(
            drop=True
        )
        y = dev_df["requester_received_pizza"].values

        # Concatenate Embeddings (Vertical Stack)
        # Anchor: [N_train + N_val, 768]
        dev_anchor = np.vstack([embeddings["train_anchor"], embeddings["val_anchor"]])
        # Aux: [N_train + N_val, 768]
        dev_aux = np.vstack([embeddings["train_aux"], embeddings["val_aux"]])

        # Test Embeddings
        test_anchor = embeddings["test_anchor"]
        test_aux = embeddings["test_aux"]

        self.logger.info(f"Combined Development Set Size: {len(dev_df)}")
        self.logger.info(f"Test Set Size: {len(test_df)}")

        # 4. Initialize CV Containers
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        oof_preds = np.zeros(len(dev_df))
        test_preds_accumulator = np.zeros(len(test_df))
        fold_scores = []

        # 5. Cross-Validation Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, y)):
            self.logger.info(
                f"================ Fold {fold + 1} / {self.n_folds} ================"
            )

            # --- Data Splitting ---
            # Metadata
            X_train_meta = dev_df.iloc[train_idx]
            X_val_meta = dev_df.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Embeddings
            X_train_anchor = dev_anchor[train_idx]
            X_val_anchor = dev_anchor[val_idx]

            X_train_aux = dev_aux[train_idx]
            X_val_aux = dev_aux[val_idx]

            # --- Feature Processing (Pipeline) ---
            # Instantiate a fresh pipeline to avoid leakage
            pipeline = WhitenedFusionPipeline()

            # Fit ONLY on training data of this fold
            pipeline.fit(X_train_anchor, X_train_aux, X_train_meta)

            # Transform all sets
            X_train_fused = pipeline.transform(
                X_train_anchor, X_train_aux, X_train_meta
            )
            X_val_fused = pipeline.transform(X_val_anchor, X_val_aux, X_val_meta)
            X_test_fused = pipeline.transform(test_anchor, test_aux, test_df)

            # --- Model Training ---
            builder = ModelBuilder()
            optimizer = builder.get_bagged_lr_optimizer()

            self.logger.info(f"Fitting model for Fold {fold + 1}...")
            optimizer.fit(X_train_fused, y_train)

            self.logger.info(
                f"Best Parameters (Fold {fold + 1}): {optimizer.best_params_}"
            )

            # --- Validation Inference ---
            # Predict probabilities (class 1)
            val_probs = optimizer.predict_proba(X_val_fused)[:, 1]
            oof_preds[val_idx] = val_probs

            # Calculate Fold Metric
            fold_auc = roc_auc_score(y_val, val_probs)
            fold_scores.append(fold_auc)
            self.logger.info(f"Fold {fold + 1} ROC AUC: {fold_auc}")

            # --- Test Inference ---
            # Predict on test set using this fold's model
            test_probs = optimizer.predict_proba(X_test_fused)[:, 1]
            test_preds_accumulator += test_probs

        # 6. Final Evaluation & Submission
        # Average test predictions across folds
        avg_test_preds = test_preds_accumulator / self.n_folds

        # Calculate Overall OOF Score
        overall_auc = roc_auc_score(y, oof_preds)
        avg_fold_auc = np.mean(fold_scores)

        self.logger.info("================ Training Complete ================")
        self.logger.info(f"Average Fold ROC AUC: {avg_fold_auc}")
        self.logger.info(f"Overall OOF ROC AUC:  {overall_auc}")

        # Save Submission
        self._save_submission(test_df, avg_test_preds)

    def _save_submission(self, test_df, predictions):
        """
        Saves the test predictions to the submission file.
        """
        submission_path = Config.SUBMISSION_FILE_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": predictions,
            }
        )

        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved successfully to: {submission_path}")
