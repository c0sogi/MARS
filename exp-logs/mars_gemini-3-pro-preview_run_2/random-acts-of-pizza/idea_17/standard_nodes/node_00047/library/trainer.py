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
        """Returns paths for saving/loading model and transformer for a specific fold."""
        model_path = os.path.join(self.models_dir, f"model_fold_{fold_idx}.joblib")
        transformer_path = os.path.join(
            self.models_dir, f"transformer_fold_{fold_idx}.joblib"
        )
        return model_path, transformer_path

    def load_all_data(self, split="train"):
        """
        Loads tabular data and embeddings for a given split.
        Returns:
            X_req (np.ndarray): Request embeddings
            X_meta (np.ndarray): Numerical metadata
            y (np.ndarray or None): Labels
            ids (pd.Series): Request IDs
        """
        self.logger.info(f"Loading all data for split: {split}")

        # 1. Load Tabular Data
        df = self.data_loader.load_dataset(split=split, load_cached_data=True)

        # 2. Get Embeddings (View 1)
        # Embedder handles caching internally
        X_req = self.embedder.get_embeddings(
            df, split=split, view_type="request", load_cached_data=True
        )

        # 3. Get Metadata (View 3)
        # Extract numerical columns defined in DataLoader
        meta_cols = self.data_loader.numerical_features
        X_meta = df[meta_cols].values

        # 4. Get Labels and IDs
        y = (
            df["requester_received_pizza"].values
            if "requester_received_pizza" in df.columns
            else None
        )
        ids = df["request_id"]

        return X_req, X_meta, y, ids

    def run_training(self):
        """
        Executes the 5-fold stratified cross-validation training loop.
        """
        set_seed(Config.SEED)

        # Load Training Data
        X_req_all, X_meta_all, y_all, _ = self.load_all_data("train")

        # Setup CV
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(y_all))
        fold_scores = []

        self.logger.info(
            f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation..."
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_req_all, y_all)):
            self.logger.info(f"\n{'='*20} Fold {fold_idx} {'='*20}")

            # Split Data
            X_req_train, X_req_val = X_req_all[train_idx], X_req_all[val_idx]
            X_meta_train, X_meta_val = X_meta_all[train_idx], X_meta_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            # 1. Feature Fusion (ViewTransformer)
            # Fit ONLY on training data to prevent leakage
            vt = ViewTransformer()
            vt.fit(X_meta_train)

            # Transform Train and Val
            X_train_fused = vt.transform(X_req_train, X_meta_train)
            X_val_fused = vt.transform(X_req_val, X_meta_val)

            # 2. Hyperparameter Tuning (Inner Loop)
            # We use a manual grid search with internal CV on the fold's training data
            grid = ParameterGrid(self.model_builder.get_hyperparameter_grid())
            best_score = -1.0
            best_params = None

            self.logger.info(f"Tuning hyperparameters for Fold {fold_idx}...")

            # Use a smaller inner CV for speed
            inner_cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=Config.SEED
            )

            for params in grid:
                # Instantiate model with current params
                model = self.model_builder.get_bagged_ensemble(
                    C=params["C"], class_weight=params["class_weight"]
                )

                # Evaluate using cross_val_score
                # Note: n_jobs=1 here because BaggingClassifier already uses n_jobs=Config.N_JOBS
                # We want to avoid over-subscription of threads.
                # However, cross_val_score can run in parallel if model n_jobs=1.
                # Given Config.N_JOBS is 12, we let the ensemble handle parallelism.
                scores = cross_val_score(
                    model,
                    X_train_fused,
                    y_train,
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=1,
                )
                avg_score = np.mean(scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params

            self.logger.info(
                f"Best Params Fold {fold_idx}: {best_params} (CV AUC: {best_score:.6f})"
            )

            # 3. Train Final Model for Fold
            final_model = self.model_builder.get_bagged_ensemble(
                C=best_params["C"], class_weight=best_params["class_weight"]
            )
            final_model.fit(X_train_fused, y_train)

            # 4. Evaluate on Validation Set
            val_probs = final_model.predict_proba(X_val_fused)[:, 1]
            fold_auc = roc_auc_score(y_val, val_probs)
            self.logger.info(f"Fold {fold_idx} Validation AUC: {fold_auc:.10f}")

            oof_preds[val_idx] = val_probs
            fold_scores.append(fold_auc)

            # 5. Save Artifacts
            model_path, transformer_path = self._get_fold_paths(fold_idx)
            joblib.dump(final_model, model_path)
            joblib.dump(vt, transformer_path)
            self.logger.info(f"Saved model and transformer for Fold {fold_idx}")

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
        X_req_test, X_meta_test, _, ids_test = self.load_all_data("test")

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
            X_test_fused = vt.transform(X_req_test, X_meta_test)

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
