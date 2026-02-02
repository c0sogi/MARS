import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import seed_everything, save_pickle, load_pickle, format_submission


class DualStreamLDA(BaseEstimator, ClassifierMixin):
    """
    A custom scikit-learn estimator that implements the specific modeling strategy:
    1. Splits concatenated features into DINOv2 (Geometry) and ConvNeXt (Texture) streams.
    2. Applies Independent Subspace Reduction (PCA) to each stream to preserve distinct manifold structures.
    3. Concatenates reduced streams with tabular features.
    4. Applies Global Gaussianization (QuantileTransformer) to strictly satisfy LDA assumptions.
    5. Classifies using Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
    """

    def __init__(self, pca_variance=0.99, lda_solver="lsqr", lda_shrinkage="auto"):
        self.pca_variance = pca_variance
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage

        # Internal state
        self.pca_dino = None
        self.pca_conv = None
        self.qt = None
        self.lda = None

        # Feature dimensions (Hardcoded based on model architecture in Config)
        # DINOv2 ViT-Large: 1024, ConvNeXt Large: 1536
        self.dino_dim = 1024

    def fit(self, X_img, X_tab, y):
        """
        Fits the pipeline.
        X_img: (N, 2560) - Concatenated DINO+ConvNeXt features
        X_tab: (N, 192) - Tabular features
        y: (N,) - Labels
        """
        # 1. Split Streams
        X_dino = X_img[:, : self.dino_dim]
        X_conv = X_img[:, self.dino_dim :]

        # 2. Independent Subspace Reduction (PCA)
        self.pca_dino = PCA(n_components=self.pca_variance, random_state=Config.SEED)
        X_dino_red = self.pca_dino.fit_transform(X_dino)

        self.pca_conv = PCA(n_components=self.pca_variance, random_state=Config.SEED)
        X_conv_red = self.pca_conv.fit_transform(X_conv)

        # 3. Early Fusion
        X_combined = np.hstack([X_dino_red, X_conv_red, X_tab])

        # 4. Global Gaussianization
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )
        X_gauss = self.qt.fit_transform(X_combined)

        # 5. LDA Classifier
        self.lda = LinearDiscriminantAnalysis(
            solver=self.lda_solver, shrinkage=self.lda_shrinkage
        )
        self.lda.fit(X_gauss, y)

        return self

    def predict_proba(self, X_img, X_tab):
        """
        Predicts class probabilities.
        """
        # 1. Split Streams
        X_dino = X_img[:, : self.dino_dim]
        X_conv = X_img[:, self.dino_dim :]

        # 2. Transform Streams (PCA)
        X_dino_red = self.pca_dino.transform(X_dino)
        X_conv_red = self.pca_conv.transform(X_conv)

        # 3. Early Fusion
        X_combined = np.hstack([X_dino_red, X_conv_red, X_tab])

        # 4. Gaussianize
        X_gauss = self.qt.transform(X_combined)

        # 5. Predict
        return self.lda.predict_proba(X_gauss)


class ModelingPipeline:
    """
    Manages the Cross-Validation Ensemble and Inference.
    Handles the complexities of Manifold Densification (grouping 3 centroids per image).
    """

    def __init__(self):
        self.models = []
        self.label_encoder = LabelEncoder()
        self.n_folds = Config.N_FOLDS

    def run_training(self, train_data):
        """
        Executes Stratified K-Fold Cross-Validation on the densified dataset.
        Ensures no data leakage by splitting based on original image IDs.

        Args:
            train_data (dict): Contains 'img', 'tab', 'ids', 'labels' (densified 3x).
        """
        seed_everything()

        X_img = train_data["img"]
        X_tab = train_data["tab"]
        ids = train_data["ids"]
        y_raw = train_data["labels"]

        # Encode Labels
        # Fit on unique labels to ensure consistency
        self.label_encoder.fit(y_raw)
        y = self.label_encoder.transform(y_raw)

        # Save LabelEncoder for inference
        save_pickle(
            self.label_encoder, os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
        )

        # Prepare Stratified Split based on Unique IDs
        # The dataset contains 3 rows per image ID (Centroids A, B, C).
        # We must split unique IDs to prevent train/val leakage.
        unique_ids = np.unique(ids)

        # Map each unique ID to its label for stratification
        # (All centroids of the same ID share the same label)
        id_to_label = {}
        for i, uid in enumerate(ids):
            if uid not in id_to_label:
                id_to_label[uid] = y[i]

        unique_labels = np.array([id_to_label[uid] for uid in unique_ids])

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")
        fold_scores = []

        for fold, (train_idx_unique, val_idx_unique) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            # Identify IDs for this fold
            train_ids_fold = unique_ids[train_idx_unique]
            val_ids_fold = unique_ids[val_idx_unique]

            # Create boolean masks for the densified dataset
            train_mask = np.isin(ids, train_ids_fold)
            val_mask = np.isin(ids, val_ids_fold)

            # Split Data
            X_img_train, X_tab_train, y_train = (
                X_img[train_mask],
                X_tab[train_mask],
                y[train_mask],
            )
            X_img_val, X_tab_val, y_val = X_img[val_mask], X_tab[val_mask], y[val_mask]
            val_ids_densified = ids[val_mask]

            # Train Model
            model = DualStreamLDA(
                pca_variance=Config.PCA_VARIANCE,
                lda_solver=Config.LDA_SOLVER,
                lda_shrinkage=Config.LDA_SHRINKAGE,
            )
            model.fit(X_img_train, X_tab_train, y_train)

            # Validation Inference
            val_probs_densified = model.predict_proba(X_img_val, X_tab_val)

            # Aggregate Predictions (Average across 3 centroids per image)
            # We group predictions by ID to compute the per-image score
            id_probs_map = {}
            id_true_map = {}

            for i, uid in enumerate(val_ids_densified):
                if uid not in id_probs_map:
                    id_probs_map[uid] = []
                    id_true_map[uid] = y_val[i]
                id_probs_map[uid].append(val_probs_densified[i])

            agg_probs = []
            agg_true = []

            # Sort by ID to ensure alignment
            for uid in sorted(id_probs_map.keys()):
                # Average probability vectors for this image
                p_mean = np.mean(np.stack(id_probs_map[uid]), axis=0)
                agg_probs.append(p_mean)
                agg_true.append(id_true_map[uid])

            agg_probs = np.array(agg_probs)
            agg_true = np.array(agg_true)

            # Calculate Metric
            score = log_loss(
                agg_true, agg_probs, labels=range(len(self.label_encoder.classes_))
            )
            fold_scores.append(score)
            print(f"Fold {fold+1} Log Loss: {score:.15f}")

            # Store Model
            self.models.append(model)
            save_pickle(
                model, os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
            )

        print(f"Average Log Loss: {np.mean(fold_scores):.15f}")

    def run_inference(self, test_data):
        """
        Generates submission predictions using Full-Manifold Test-Time Aggregation.
        1. Predicts on all 3 centroids for every test image using every ensemble model.
        2. Averages predictions across models (Ensemble).
        3. Averages predictions across centroids (Manifold Aggregation).
        """
        X_img = test_data["img"]
        X_tab = test_data["tab"]
        ids = test_data["ids"]

        if not self.models:
            print("No models found in memory. Attempting to load from disk...")
            for fold in range(self.n_folds):
                path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
                if os.path.exists(path):
                    self.models.append(load_pickle(path))
            if not self.models:
                raise ValueError("No trained models available for inference.")

            # Load label encoder
            le_path = os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
            if os.path.exists(le_path):
                self.label_encoder = load_pickle(le_path)

        print("Running inference with Full-Manifold Test-Time Aggregation...")

        # 1. Ensemble Prediction on Densified Data
        # Accumulate predictions from all K folds
        ensemble_preds = []
        for model in self.models:
            preds = model.predict_proba(X_img, X_tab)
            ensemble_preds.append(preds)

        # Average across ensemble members
        # Shape: (N_densified, n_classes)
        avg_preds_densified = np.mean(ensemble_preds, axis=0)

        # 2. Manifold Aggregation (Average across centroids)
        # Group by ID and average
        unique_ids = np.unique(ids)
        final_preds = []
        final_ids = []

        # Map ID -> List of probability vectors (one for each centroid)
        id_probs_map = {uid: [] for uid in unique_ids}
        for i, uid in enumerate(ids):
            id_probs_map[uid].append(avg_preds_densified[i])

        # Compute final mean per image
        for uid in unique_ids:
            p_mean = np.mean(np.stack(id_probs_map[uid]), axis=0)
            final_preds.append(p_mean)
            final_ids.append(uid)

        final_preds = np.array(final_preds)
        final_ids = np.array(final_ids)

        # 3. Format and Save Submission
        class_names = self.label_encoder.classes_
        format_submission(final_ids, final_preds, class_names)
