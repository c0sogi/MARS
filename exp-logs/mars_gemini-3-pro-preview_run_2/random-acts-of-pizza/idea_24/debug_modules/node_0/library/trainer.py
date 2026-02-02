import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import setup_logger, compute_auc, set_seed
from library.feature_extractors import (
    TextEmbedder,
    BayesianSubredditEncoder,
    RankGaussScaler,
)
from library.models import TunedLogisticRegression


class CrossValidationStacker:
    """
    Orchestrates the Passthrough-Stacked Hybrid Linear Ensemble (PSHLE) strategy.
    Performs 5-Fold Stratified CV to train Stage 1 Experts and Stage 2 Meta-Learner.
    """

    def __init__(self):
        self.logger = setup_logger("CrossValidationStacker")
        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_SEED
        set_seed(self.random_state)

    def run_cv(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
    ):
        """
        Executes the Cross-Validation loop.

        Args:
            df_train (pd.DataFrame): Training set (from loader).
            df_val (pd.DataFrame): Validation set (from loader).
            df_test (pd.DataFrame): Test set (from loader).

        Returns:
            float: Overall OOF AUC score.
        """
        self.logger.info("Initializing Cross-Validation Stacker...")

        # 1. Merge Train and Val to maximize data for CV
        # We ignore the fixed split provided by metadata for the purpose of 5-Fold CV
        # to ensure robust evaluation and maximum training data.
        df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        y_full = df_full[Config.TARGET_COL].values

        self.logger.info(f"Merged Train+Val shape: {df_full.shape}")

        # 2. Precompute Text Embeddings
        # This is expensive, so we do it once outside the loop.
        # SBERT is frozen/unsupervised, so no target leakage.
        self.logger.info("Generating/Loading Text Embeddings...")
        embedder = TextEmbedder()

        # Cache names distinguish between full train and test
        cache_suffix = "_debug" if Config.DEBUG_SAMPLE_SIZE else ""

        X_text_full = embedder.transform(
            df_full, cache_name=f"train_full{cache_suffix}"
        )
        X_text_test = embedder.transform(df_test, cache_name=f"test{cache_suffix}")

        # 3. Initialize CV containers
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        oof_preds = np.zeros(len(df_full))
        test_preds_accum = np.zeros(len(df_test))
        fold_aucs = []

        # 4. CV Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, y_full)):
            self.logger.info(f"\n{'='*20} Fold {fold + 1} / {self.n_folds} {'='*20}")

            # --- Slicing Data ---
            # Text Embeddings
            X_text_train = X_text_full[train_idx]
            X_text_val = X_text_full[val_idx]

            # DataFrames (for History & Metadata)
            df_train_fold = df_full.iloc[train_idx].copy()
            df_val_fold = df_full.iloc[val_idx].copy()

            # Targets
            y_train_fold = y_full[train_idx]
            y_val_fold = y_full[val_idx]

            # --- Stage 1: Text Expert ---
            self.logger.info(
                "[Stage 1] Training Text Expert (SBERT + LogisticRegression)..."
            )
            text_model = TunedLogisticRegression(
                param_grid=Config.TEXT_EXPERT_GRID, random_state=self.random_state
            )
            text_model.fit(X_text_train, y_train_fold)

            # Get Probabilities (Class 1)
            prob_text_train = text_model.predict_proba(X_text_train)[:, 1]
            prob_text_val = text_model.predict_proba(X_text_val)[:, 1]
            prob_text_test = text_model.predict_proba(X_text_test)[:, 1]

            # --- Stage 1: History Expert ---
            self.logger.info(
                "[Stage 1] Training History Expert (Bayesian Target Encoding)..."
            )
            history_encoder = BayesianSubredditEncoder()
            # Must pass y as Series for internal .values/.mean() calls
            history_encoder.fit(df_train_fold, pd.Series(y_train_fold))

            score_hist_train = history_encoder.transform(df_train_fold)
            score_hist_val = history_encoder.transform(df_val_fold)
            score_hist_test = history_encoder.transform(df_test)

            # --- Preprocessing: Metadata ---
            self.logger.info("[Stage 1] Scaling Metadata (RankGauss)...")
            scaler = RankGaussScaler()
            scaler.fit(df_train_fold)

            meta_train = scaler.transform(df_train_fold)
            meta_val = scaler.transform(df_val_fold)
            meta_test = scaler.transform(df_test)

            # --- Fusion (Passthrough Stacking) ---
            # Combine: [Text_Prob, History_Score, Metadata_Scaled]
            def fuse_features(probs, hist, meta):
                return np.hstack([probs.reshape(-1, 1), hist, meta])  # Already (N, 1)

            X_meta_train = fuse_features(prob_text_train, score_hist_train, meta_train)
            X_meta_val = fuse_features(prob_text_val, score_hist_val, meta_val)
            X_meta_test = fuse_features(prob_text_test, score_hist_test, meta_test)

            # --- Stage 2: Meta-Learner ---
            self.logger.info(
                f"[Stage 2] Training Meta-Learner (Stacked LogisticRegression) on shape {X_meta_train.shape}..."
            )
            meta_model = TunedLogisticRegression(
                param_grid=Config.META_LEARNER_GRID, random_state=self.random_state
            )
            meta_model.fit(X_meta_train, y_train_fold)

            # --- Inference ---
            val_preds_fold = meta_model.predict_proba(X_meta_val)[:, 1]
            test_preds_fold = meta_model.predict_proba(X_meta_test)[:, 1]

            # Store OOF
            oof_preds[val_idx] = val_preds_fold

            # Accumulate Test Predictions
            test_preds_accum += test_preds_fold

            # Evaluate Fold
            fold_auc = compute_auc(y_val_fold, val_preds_fold)
            fold_aucs.append(fold_auc)
            self.logger.info(f"Fold {fold + 1} AUC: {fold_auc}")

        # 5. Final Evaluation & Submission
        overall_auc = compute_auc(y_full, oof_preds)
        mean_auc = np.mean(fold_aucs)

        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"CV Complete.")
        self.logger.info(f"Mean Fold AUC: {mean_auc}")
        self.logger.info(f"Overall OOF AUC: {overall_auc}")
        self.logger.info(f"{'='*40}")

        # Average Test Predictions
        avg_test_preds = test_preds_accum / self.n_folds

        self._save_submission(df_test, avg_test_preds)

        return overall_auc

    def _save_submission(self, df_test, preds):
        """
        Saves the submission file in the required format.
        """
        self.logger.info("Generating submission file...")

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission = pd.DataFrame(
            {"request_id": df_test[Config.ID_COL], Config.TARGET_COL: preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to: {Config.SUBMISSION_PATH}")
