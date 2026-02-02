import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import itertools

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_dataset
from library.feature_text import SBERTEmbedder
from library.feature_topic import LDATopicExtractor
from library.feature_meta import MetadataScaler
from library.model_builder import create_classifier


class ModelEngine:
    """
    Orchestrates the Topic-Augmented Dense Fusion (TADF) pipeline.
    Handles data loading, feature fusion, stratified cross-validation,
    hyperparameter tuning, and submission generation.
    """

    def __init__(self):
        self.logger = setup_logger("model_engine")
        self.embedder = SBERTEmbedder()
        set_seed(Config.RANDOM_SEED)

    def run(self, load_cached_data: bool = True):
        """
        Executes the full training and inference pipeline.

        Args:
            load_cached_data (bool): Whether to use cached intermediate files.
        """
        self.logger.info("Starting TADF Model Engine...")

        # ----------------------------------------------------------------------
        # 1. Load Data
        # ----------------------------------------------------------------------
        # Load disjoint splits as defined by metadata
        df_train_split, df_val_split, df_test = load_dataset(
            load_cached_data=load_cached_data
        )

        # Combine train and val to perform our own Stratified K-Fold CV
        # Reset index is crucial for correct iloc slicing later
        df_full_train = pd.concat([df_train_split, df_val_split], ignore_index=True)
        y_full = df_full_train["requester_received_pizza"].values.astype(int)

        self.logger.info(f"Full training set shape: {df_full_train.shape}")
        self.logger.info(f"Test set shape: {df_test.shape}")

        # ----------------------------------------------------------------------
        # 2. Pre-compute Static Embeddings (View 1: Semantic Content)
        # ----------------------------------------------------------------------
        # We compute these once as they are independent of the fold split (SBERT is pre-trained/frozen)
        self.logger.info("Retrieving SBERT embeddings...")

        emb_train_part = self.embedder.process_and_cache(
            df_train_split, Config.TRAIN_EMBEDDINGS_PATH, load_cached_data
        )
        emb_val_part = self.embedder.process_and_cache(
            df_val_split, Config.VAL_EMBEDDINGS_PATH, load_cached_data
        )
        emb_test = self.embedder.process_and_cache(
            df_test, Config.TEST_EMBEDDINGS_PATH, load_cached_data
        )

        # Stack train and val embeddings to match df_full_train
        X_text_full = np.vstack([emb_train_part, emb_val_part])

        # ----------------------------------------------------------------------
        # 3. Stratified Cross-Validation Loop
        # ----------------------------------------------------------------------
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        # Array to store accumulated test predictions (for averaging)
        test_preds_sum = np.zeros(len(df_test))

        # Store out-of-fold predictions for global evaluation
        oof_preds = np.zeros(len(df_full_train))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_full, y_full)):
            self.logger.info(f"=== Starting Fold {fold + 1}/{Config.N_FOLDS} ===")

            # --- A. Data Slicing ---
            # Text Embeddings (View 1)
            X_text_train = X_text_full[train_idx]
            X_text_val = X_text_full[val_idx]

            # DataFrame slices for Dynamic Features
            df_fold_train = df_full_train.iloc[train_idx]
            df_fold_val = df_full_train.iloc[val_idx]
            y_fold_train = y_full[train_idx]
            y_fold_val = y_full[val_idx]

            # --- B. Dynamic Feature Extraction (Fit on Train, Transform Val/Test) ---

            # View 2: User Persona (LDA)
            lda_extractor = LDATopicExtractor(
                n_components=Config.LDA_N_COMPONENTS,
                min_df=Config.LDA_MIN_DF,
                random_state=Config.LDA_RANDOM_STATE,
            )
            X_topic_train = lda_extractor.fit_transform(
                df_fold_train["subreddit_list_str"]
            )
            X_topic_val = lda_extractor.transform(df_fold_val["subreddit_list_str"])
            X_topic_test = lda_extractor.transform(df_test["subreddit_list_str"])

            # View 3: Robust Metadata (RankGauss)
            meta_scaler = MetadataScaler(random_state=Config.RANDOM_SEED)
            X_meta_train = meta_scaler.fit_transform(df_fold_train)
            X_meta_val = meta_scaler.transform(df_fold_val)
            X_meta_test = meta_scaler.transform(df_test)

            # --- C. Feature Fusion ---
            X_train_fold = np.hstack([X_text_train, X_topic_train, X_meta_train])
            X_val_fold = np.hstack([X_text_val, X_topic_val, X_meta_val])
            X_test_fold = np.hstack([emb_test, X_topic_test, X_meta_test])

            # --- D. Hyperparameter Tuning (Grid Search) ---
            best_auc = -1.0
            best_model = None
            best_params = {}

            # Create parameter combinations
            keys = Config.PARAM_GRID.keys()
            values = Config.PARAM_GRID.values()
            param_combinations = [
                dict(zip(keys, v)) for v in itertools.product(*values)
            ]

            for params in param_combinations:
                # Create and train model
                clf = create_classifier(
                    C=params["C"],
                    class_weight=params["class_weight"],
                    n_estimators=Config.BAGGING_N_ESTIMATORS,
                    random_state=Config.RANDOM_SEED,
                )

                clf.fit(X_train_fold, y_fold_train)

                # Evaluate
                y_pred_val = clf.predict_proba(X_val_fold)[:, 1]
                auc = roc_auc_score(y_fold_val, y_pred_val)

                if auc > best_auc:
                    best_auc = auc
                    best_model = clf
                    best_params = params

            self.logger.info(f"Fold {fold + 1} Best AUC: {best_auc}")
            self.logger.info(f"Fold {fold + 1} Best Params: {best_params}")

            # --- E. Inference ---
            # Generate predictions for Validation set (for OOF) using best model
            oof_preds[val_idx] = best_model.predict_proba(X_val_fold)[:, 1]

            # Generate predictions for Test set using best model
            fold_test_preds = best_model.predict_proba(X_test_fold)[:, 1]
            test_preds_sum += fold_test_preds

        # ----------------------------------------------------------------------
        # 4. Global Evaluation & Submission
        # ----------------------------------------------------------------------
        global_auc = roc_auc_score(y_full, oof_preds)
        self.logger.info(f"Overall OOF AUC: {global_auc}")

        # Average test predictions
        avg_test_preds = test_preds_sum / Config.N_FOLDS

        self.logger.info("Generating submission file...")
        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        Config.ensure_directories()
        submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
