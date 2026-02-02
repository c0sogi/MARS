import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.data_processing import DensificationManager
from library.model_definition import create_hybrid_pipeline


class ModelExecutor:
    """
    Orchestrates the training and inference process using the Selective-Topology
    Orthogonal Manifold-Densified LDA strategy.
    """

    def __init__(self):
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "execution.log"))
        self.densification_manager = DensificationManager()
        seed_everything(Config.SEED)
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def _prepare_feature_matrix(self, dino, conv, tab):
        """
        Concatenates the three feature streams into a single matrix.
        Order: [DINO (Linear) | ConvNeXt (Linear) | Tabular (Non-Linear)]
        """
        return np.hstack([dino, conv, tab])

    def train_ensemble(self, load_cached_data=True):
        """
        Trains a Stratified K-Fold ensemble of Hybrid LDA pipelines.

        Args:
            load_cached_data (bool): Whether to load densified features from cache.

        Returns:
            list: List of trained pipeline objects.
        """
        self.logger.info("Starting ensemble training...")

        # 1. Load Densified Training Data
        # Returns: ids (3N,), dino (3N, D1), conv (3N, D2), tab (3N, 192), labels (3N,)
        ids, dino, conv, tab, labels = self.densification_manager.prepare_training_data(
            load_cached_data=load_cached_data
        )

        # Construct full feature matrix
        X = self._prepare_feature_matrix(dino, conv, tab)
        y = labels

        # Get dimensions for the pipeline definition
        dino_dim = dino.shape[1]
        conv_dim = conv.shape[1]
        tab_dim = tab.shape[1]

        self.logger.info(f"Feature Matrix Shape: {X.shape}")

        # 2. Setup Stratified K-Fold
        # CRITICAL: Split based on UNIQUE original IDs to prevent data leakage.
        # Each image has 3 centroids in the densified dataset. All 3 must be in the same fold.
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        unique_labels = y[unique_indices]

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []
        pipelines = []

        for fold, (train_idx_unique, val_idx_unique) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            self.logger.info(f"Training Fold {fold + 1}/{Config.N_FOLDS}...")

            # Identify IDs for this fold
            train_ids_fold = unique_ids[train_idx_unique]
            val_ids_fold = unique_ids[val_idx_unique]

            # Create boolean masks to select all centroids corresponding to these IDs
            train_mask = np.isin(ids, train_ids_fold)
            val_mask = np.isin(ids, val_ids_fold)

            # Split data
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # 3. Create and Fit Pipeline
            # Pipeline: [PCA(Visual) + Quantile(Tabular)] -> LDA
            pipeline = create_hybrid_pipeline(dino_dim, conv_dim, tab_dim)
            pipeline.fit(X_train, y_train)

            # 4. Evaluate
            # Predict probabilities on validation set
            y_pred_proba = pipeline.predict_proba(X_val)

            # Calculate Log Loss
            score = log_loss(y_val, y_pred_proba, labels=pipeline.classes_)
            fold_scores.append(score)
            self.logger.info(f"Fold {fold + 1} Log Loss: {score:.15f}")

            # 5. Save Model
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)
            pipelines.append(pipeline)

        avg_score = np.mean(fold_scores)
        self.logger.info(f"Average CV Log Loss: {avg_score:.15f}")

        return pipelines

    def generate_submission(self, pipelines, load_cached_data=True):
        """
        Generates predictions for the test set using the trained ensemble.
        Performs aggregation across the 3 centroids per image.

        Args:
            pipelines (list): List of trained pipelines.
            load_cached_data (bool): Whether to load densified features from cache.
        """
        self.logger.info("Generating submission...")

        # 1. Load Densified Test Data
        ids, dino, conv, tab = self.densification_manager.prepare_inference_data(
            load_cached_data=load_cached_data
        )

        X_test = self._prepare_feature_matrix(dino, conv, tab)

        # 2. Ensemble Inference
        # Get class names from the first pipeline
        classes = pipelines[0].classes_
        n_classes = len(classes)
        n_samples = len(X_test)

        # Accumulator for probabilities
        ensemble_proba = np.zeros((n_samples, n_classes))

        for i, pipeline in enumerate(pipelines):
            # Predict on all centroids
            proba = pipeline.predict_proba(X_test)
            ensemble_proba += proba

        # Average across ensemble members
        ensemble_proba /= len(pipelines)

        # 3. Aggregation (Centroids -> Image)
        # Group by ID and compute mean probability across the 3 centroids
        df_pred = pd.DataFrame(ensemble_proba, columns=classes)
        df_pred["id"] = ids

        # Groupby 'id' sorts the IDs, which is fine
        df_submission = df_pred.groupby("id").mean().reset_index()

        # 4. Save Submission
        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

        return df_submission

    def run(self):
        """
        Executes the full pipeline: Training -> Inference.
        """
        pipelines = self.train_ensemble(load_cached_data=True)
        self.generate_submission(pipelines, load_cached_data=True)
