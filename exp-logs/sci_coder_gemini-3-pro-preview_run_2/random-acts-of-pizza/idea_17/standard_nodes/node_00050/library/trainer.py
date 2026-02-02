import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid, cross_val_score
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import DataLoader
from library.feature_engineering import Embedder, ViewTransformer
from library.model_builder import ModelBuilder


class Trainer:
    """
    Orchestrates the training and inference pipeline for Idea 17: AMBLE.
    Handles data loading, cross-validation, hyperparameter tuning, model persistence, and submission.
    """

    def __init__(self):
        self.logger = setup_logger("Trainer")
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize components
        self.data_loader = DataLoader()
        self.embedder = Embedder()
        self.model_builder = ModelBuilder()

    def _get_fold_paths(self, fold_idx):
        """Returns paths for saving/loading model for a specific fold."""
        # We only save the pipeline now, which contains both transformer and model
        model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold_idx}.joblib")
        return model_path

    def load_all_data(self, split="train"):
        """
        Loads tabular data and embeddings for a given split.
        Concatenates them into a single matrix X for Pipeline usage.
        """
        self.logger.info(f"Loading all data for split: {split}")

        # 1. Load Tabular Data
        df = self.data_loader.load_dataset(split=split, load_cached_data=True)

        # 2. Get Embeddings (View 1 & 2)
        X_req = self.embedder.get_embeddings(
            df, split=split, view_type="request", load_cached_data=True
        )
        X_hist = self.embedder.get_embeddings(
            df, split=split, view_type="history", load_cached_data=True
        )

        # 3. Get Metadata (View 3)
        meta_cols = self.data_loader.numerical_features
        X_meta = df[meta_cols].values

        # 4. Concatenate
        # Order must match ViewTransformer logic: [Req, Hist, Meta]
        X_all = np.hstack([X_req, X_hist, X_meta])

        # 5. Get Labels and IDs
        y = (
            df["requester_received_pizza"].values
            if "requester_received_pizza" in df.columns
            else None
        )
        ids = df["request_id"]

        return X_all, y, ids

    def run_training(self):
        """
        Executes the 5-fold stratified cross-validation training loop.
        """
        set_seed(Config.SEED)

        # Load Training Data
        X_all, y_all, _ = self.load_all_data("train")

        # Setup CV
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(y_all))
        fold_scores = []

        self.logger.info(
            f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation..."
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            self.logger.info(f"\n{'='*20} Fold {fold_idx} {'='*20}")

            # Split Data
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            # Hyperparameter Tuning (Inner Loop)
            # We use a manual grid search but with a Pipeline to ensure correct stats calculation
            grid = ParameterGrid(self.model_builder.get_hyperparameter_grid())
            best_score = -1.0
            best_params = None

            self.logger.info(f"Tuning hyperparameters for Fold {fold_idx}...")

            # Use a smaller inner CV for speed
            inner_cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=Config.SEED
            )

            for params in grid:
                # Instantiate pipeline
                pipeline = self.model_builder.get_pipeline()
                pipeline.set_params(**params)

                # Evaluate using cross_val_score
                # The Pipeline ensures ViewTransformer.fit() is called on inner train sets
                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=1,  # Bagging inside pipeline handles parallelism
                )
                avg_score = np.mean(scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params

            self.logger.info(
                f"Best Params Fold {fold_idx}: {best_params} (CV AUC: {best_score:.6f})"
            )

            # Train Final Model for Fold
            final_pipeline = self.model_builder.get_pipeline()
            final_pipeline.set_params(**best_params)
            final_pipeline.fit(X_train, y_train)

            # Evaluate on Validation Set
            val_probs = final_pipeline.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, val_probs)
            self.logger.info(f"Fold {fold_idx} Validation AUC: {fold_auc:.10f}")

            oof_preds[val_idx] = val_probs
            fold_scores.append(fold_auc)

            # Save Artifacts
            model_path = self._get_fold_paths(fold_idx)
            joblib.dump(final_pipeline, model_path)
            self.logger.info(f"Saved pipeline for Fold {fold_idx}")

        # Summary
        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)
        overall_auc = roc_auc_score(y_all, oof_preds)

        self.logger.info(f"\n{'='*20} Summary {'='*20}")
        self.logger.info(f"Mean Fold AUC: {mean_auc:.10f} ± {std_auc:.10f}")
        self.logger.info(f"Overall OOF AUC: {overall_auc:.10f}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the ensemble of trained fold models.
        Averages the probabilities (Bagging).
        """
        self.logger.info("\nGenerating submission...")

        # Load Test Data
        X_req_test, X_hist_test, X_meta_test, _, ids_test = self.load_all_data("test")

        test_preds_sum = np.zeros(len(ids_test))

        # Iterate through folds
        for fold_idx in range(Config.N_FOLDS):
            model_path, transformer_path = self._get_fold_paths(fold_idx)

            if not os.path.exists(model_path) or not os.path.exists(transformer_path):
                raise FileNotFoundError(
                    f"Artifacts for Fold {fold_idx} not found. Run training first."
                )

            # Load artifacts
            model = joblib.load(model_path)
            vt = joblib.load(transformer_path)

            # Transform Test Data using the fold-specific transformer
            # This applies the specific PCA projection and Quantile mapping learned in that fold
            X_test_fused = vt.transform(X_req_test, X_hist_test, X_meta_test)

            # Predict
            probs = model.predict_proba(X_test_fused)[:, 1]
            test_preds_sum += probs

            self.logger.info(f"Processed Fold {fold_idx} inference.")

        # Average predictions
        avg_preds = test_preds_sum / Config.N_FOLDS

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": ids_test, "requester_received_pizza": avg_preds}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
