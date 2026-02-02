import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

from library.config import Config
from library.utils import (
    setup_logger,
    set_seed,
    ensure_directory,
    save_joblib,
    load_joblib,
)
from library.data_loader import DataLoader
from library.embedding_engine import EmbeddingEngine
from library.feature_engineer import CoherenceFeatureProcessor
from library.model_factory import ModelFactory


class ExecutionManager:
    """
    Orchestrates the Coherence-Augmented Multi-Field Asymmetric Dual-Backbone Ensemble (CM-ADBE).
    Manages data loading, embedding generation, stratified CV training, and ensemble inference.
    """

    def __init__(self):
        self.logger = setup_logger(
            "ExecutionManager", os.path.join(Config.WORKING_DIR, "execution.log")
        )
        set_seed(Config.SEED)

    def run_cv_and_inference(self, debug_sample_size=None, load_cached_data=True):
        """
        Executes the full pipeline:
        1. Loads and combines train/val data.
        2. Generates/Loads embeddings.
        3. Runs 5-Fold Stratified CV with leakage-free feature engineering.
        4. Trains Bagged Logistic Regression Ensembles.
        5. Performs CV-Bagging Inference on Test Set.
        6. Generates Submission.

        Args:
            debug_sample_size (int, optional): Number of rows to use for debugging.
            load_cached_data (bool): Whether to use cached embeddings.
        """
        self.logger.info("Starting CM-ADBE Pipeline Execution...")

        # ==========================================
        # 1. Data Loading & Preparation
        # ==========================================
        loader = DataLoader()

        # Load metadata-defined splits
        self.logger.info("Loading data splits...")
        df_train_part = loader.load_split("train")
        df_val_part = loader.load_split("val")

        # Combine into a single development set for 5-Fold CV
        df_full_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

        # Load Test Set
        df_test = loader.load_split("test")

        # Apply Debugging Sampling
        if debug_sample_size:
            self.logger.info(f"DEBUG MODE: Subsampling {debug_sample_size} rows.")
            df_full_train = df_full_train.head(debug_sample_size)
            df_test = df_test.head(debug_sample_size)

        self.logger.info(f"Full Development Set Shape: {df_full_train.shape}")
        self.logger.info(f"Test Set Shape: {df_test.shape}")

        # ==========================================
        # 2. Embedding Generation
        # ==========================================
        # We redirect cache paths to avoid conflicts with single-split caches
        # and to ensure we cache the combined dataset embeddings correctly.
        Config.CACHE_TRAIN_TITLE_MINILM = os.path.join(
            Config.WORKING_DIR, "combined_train_title_minilm.npy"
        )
        Config.CACHE_TRAIN_BODY_MINILM = os.path.join(
            Config.WORKING_DIR, "combined_train_body_minilm.npy"
        )
        Config.CACHE_TRAIN_GLOBAL_MPNET = os.path.join(
            Config.WORKING_DIR, "combined_train_global_mpnet.npy"
        )

        embedder = EmbeddingEngine()

        # Generate/Load Train Embeddings
        train_title_emb, train_body_emb, train_global_emb = (
            embedder.generate_train_embeddings(
                df_full_train, load_cached_data=load_cached_data
            )
        )

        # Generate/Load Test Embeddings
        test_title_emb, test_body_emb, test_global_emb = (
            embedder.generate_test_embeddings(
                df_test, load_cached_data=load_cached_data
            )
        )

        # Extract Metadata Features (Numerical)
        X_meta_train = df_full_train[Config.NUMERIC_COLS].fillna(0).values
        X_meta_test = df_test[Config.NUMERIC_COLS].fillna(0).values

        # Target
        y_train_full = df_full_train["requester_received_pizza"].values

        # ==========================================
        # 3. Stratified K-Fold Cross-Validation
        # ==========================================
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []
        models_dir = os.path.join(Config.WORKING_DIR, "models")
        ensure_directory(models_dir)

        self.logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified CV...")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df_full_train, y_train_full)
        ):
            self.logger.info(f"Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            # Split Data
            # Embeddings
            f_train_title, f_val_title = (
                train_title_emb[train_idx],
                train_title_emb[val_idx],
            )
            f_train_body, f_val_body = (
                train_body_emb[train_idx],
                train_body_emb[val_idx],
            )
            f_train_global, f_val_global = (
                train_global_emb[train_idx],
                train_global_emb[val_idx],
            )

            # Metadata
            f_train_meta, f_val_meta = X_meta_train[train_idx], X_meta_train[val_idx]

            # Targets
            f_y_train, f_y_val = y_train_full[train_idx], y_train_full[val_idx]

            # Feature Engineering (Fit on Train ONLY to prevent leakage)
            processor = CoherenceFeatureProcessor()
            processor.fit(f_train_title, f_train_body, f_train_global, f_train_meta)

            # Transform Train and Val
            X_train_fused = processor.transform(
                f_train_title, f_train_body, f_train_global, f_train_meta
            )
            X_val_fused = processor.transform(
                f_val_title, f_val_body, f_val_global, f_val_meta
            )

            # Model Training
            factory = ModelFactory()
            model = factory.optimize_and_train(X_train_fused, f_y_train)

            # Validation
            y_pred_val = model.predict_proba(X_val_fused)[:, 1]
            score = roc_auc_score(f_y_val, y_pred_val)
            fold_scores.append(score)

            self.logger.info(f"Fold {fold + 1} ROC AUC: {score}")

            # Save Model and Processor for Inference
            save_joblib(model, os.path.join(models_dir, f"model_fold_{fold}.joblib"))
            save_joblib(
                processor, os.path.join(models_dir, f"processor_fold_{fold}.joblib")
            )

        avg_score = np.mean(fold_scores)
        self.logger.info(f"Average CV ROC AUC: {avg_score}")
        self.logger.info(f"Fold Scores: {fold_scores}")

        # ==========================================
        # 4. Inference (CV-Bagging)
        # ==========================================
        self.logger.info("Starting Inference on Test Set (CV-Bagging)...")

        test_preds_accum = np.zeros(len(df_test))

        for fold in range(Config.N_FOLDS):
            # Load artifacts
            model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
            proc_path = os.path.join(models_dir, f"processor_fold_{fold}.joblib")

            model = load_joblib(model_path)
            processor = load_joblib(proc_path)

            # Transform Test Data using fold-specific processor
            X_test_fused = processor.transform(
                test_title_emb, test_body_emb, test_global_emb, X_meta_test
            )

            # Predict
            preds = model.predict_proba(X_test_fused)[:, 1]
            test_preds_accum += preds

        # Average predictions
        final_preds = test_preds_accum / Config.N_FOLDS

        # ==========================================
        # 5. Submission Generation
        # ==========================================
        self.logger.info("Generating Submission File...")

        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": final_preds,
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info("Pipeline Execution Completed Successfully.")
