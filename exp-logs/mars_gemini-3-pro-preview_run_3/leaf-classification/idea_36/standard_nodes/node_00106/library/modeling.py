import os
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import save_pickle, load_pickle


def create_hybrid_pipeline(dino_dim, conv_dim, tab_dim, pca_variance=0.99):
    """
    Constructs the hybrid pipeline with Selective Feature Topology.

    Applies Independent Subspace Reduction (Linear PCA) to visual streams and
    Non-Linear Gaussianization (QuantileTransformer) to tabular features.

    Args:
        dino_dim (int): Number of DINO features (column 0 to dino_dim).
        conv_dim (int): Number of ConvNeXt features.
        tab_dim (int): Number of tabular features.
        pca_variance (float): Variance to retain in PCA.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Define column slices based on concatenation order: DINO, Conv, Tabular
    # Slice objects work directly with numpy arrays in ColumnTransformer
    slice_dino = slice(0, dino_dim)
    slice_conv = slice(dino_dim, dino_dim + conv_dim)
    slice_tab = slice(dino_dim + conv_dim, dino_dim + conv_dim + tab_dim)

    # Define transformers
    # Visual streams: Linear PCA only (Independent Subspace Reduction)
    # We strictly preserve the linear topology of the visual streams.
    pca_dino = PCA(n_components=pca_variance)
    pca_conv = PCA(n_components=pca_variance)

    # Tabular stream: Non-linear Gaussianization
    # Aligns arbitrary histograms with the Gaussian assumption of LDA.
    quant_tab = QuantileTransformer(output_distribution="normal")

    # Feature Processing Block
    preprocessor = ColumnTransformer(
        transformers=[
            ("pca_dino", pca_dino, slice_dino),
            ("pca_conv", pca_conv, slice_conv),
            ("quant_tab", quant_tab, slice_tab),
        ],
        verbose_feature_names_out=False,
    )

    # Classifier Block
    # 1. Global StandardScaler: Ensures Ledoit-Wolf shrinkage applies uniformly.
    # 2. LDA with Ledoit-Wolf shrinkage (solver='lsqr', shrinkage='auto').
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    return pipeline


class LDATrainer:
    """
    Manages the training and inference of the LDA Ensemble using Stratified K-Fold.
    Handles the Convex-Densified dataset structure.
    """

    def __init__(self, config: Config):
        self.config = config
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.classes_path = os.path.join(self.models_dir, "classes.pkl")

    def _prepare_matrix(self, data_dict):
        """Concatenates feature arrays into a single matrix X."""
        # Order must match the slices in create_hybrid_pipeline: DINO, Conv, Tabular
        return np.hstack([data_dict["dino"], data_dict["conv"], data_dict["tab"]])

    def train(self, train_data):
        """
        Trains the Stratified K-Fold Ensemble on the densified dataset.

        Args:
            train_data (dict): Densified training data containing 'ids', 'dino', 'conv', 'tab', 'labels'.
        """
        logging.info("Preparing training data...")

        # 1. Prepare X and y
        X = self._prepare_matrix(train_data)
        y_raw = train_data["labels"]
        ids = train_data["ids"]

        # 2. Encode Labels
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Save classes for inference
        save_pickle(le.classes_, self.classes_path)
        logging.info(f"Saved class encoding to {self.classes_path}")

        # 3. Stratified K-Fold (Leakage-Free)
        # The densified dataset contains 6 centroids per original image.
        # We must split based on original images to ensure all centroids of an image
        # end up in the same fold.

        # Recover original IDs and Labels (taking every 6th sample)
        orig_ids = ids[::6]
        orig_labels = y[::6]

        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED
        )

        scores = []

        # Get feature dimensions for pipeline creation
        dino_dim = train_data["dino"].shape[1]
        conv_dim = train_data["conv"].shape[1]
        tab_dim = train_data["tab"].shape[1]

        logging.info(f"Starting {self.config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx_orig, val_idx_orig) in enumerate(
            skf.split(orig_ids, orig_labels)
        ):
            # Expand indices to densified dataset
            # Each original index i corresponds to indices [6*i, ..., 6*i+5] in densified data
            train_idx_dense = (train_idx_orig[:, None] * 6 + np.arange(6)).ravel()
            val_idx_dense = (val_idx_orig[:, None] * 6 + np.arange(6)).ravel()

            X_train, y_train = X[train_idx_dense], y[train_idx_dense]
            X_val, y_val = X[val_idx_dense], y[val_idx_dense]

            # Create Pipeline
            pipeline = create_hybrid_pipeline(
                dino_dim, conv_dim, tab_dim, pca_variance=self.config.PCA_VARIANCE
            )

            # Fit
            pipeline.fit(X_train, y_train)

            # Predict (Probabilities) on validation set
            probs_dense = pipeline.predict_proba(X_val)

            # Aggregate predictions for validation scoring
            # Reshape to (N_val_orig, 6, N_classes) and mean over axis 1 (centroids)
            probs_agg = probs_dense.reshape(len(val_idx_orig), 6, -1).mean(axis=1)

            # Calculate Score (Multi-class Log Loss)
            # Compare against original labels
            y_val_orig = orig_labels[val_idx_orig]
            fold_score = log_loss(y_val_orig, probs_agg)
            scores.append(fold_score)

            logging.info(f"Fold {fold} Log Loss: {fold_score}")

            # Save Model
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            save_pickle(pipeline, model_path)

        avg_score = np.mean(scores)
        logging.info(f"Average CV Log Loss: {avg_score}")

    def predict(self, test_data):
        """
        Generates predictions for the test set using the trained ensemble.
        Aggregates predictions across all 6 centroids and all K folds.

        Args:
            test_data (dict): Densified test data.

        Returns:
            pd.DataFrame: DataFrame with 'id' and class probabilities.
        """
        logging.info("Loading models and generating predictions...")

        # Load classes
        classes = load_pickle(self.classes_path)

        # Prepare X
        X = self._prepare_matrix(test_data)
        ids = test_data["ids"]

        # Initialize accumulator for probabilities
        # Shape: (N_densified_samples, N_classes)
        ensemble_probs = np.zeros((X.shape[0], len(classes)))

        # Iterate over folds
        for fold in range(self.config.N_FOLDS):
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            pipeline = load_pickle(model_path)

            # Predict
            probs = pipeline.predict_proba(X)
            ensemble_probs += probs

        # Average over folds
        ensemble_probs /= self.config.N_FOLDS

        # Create DataFrame for Aggregation
        # We need to average over the 6 centroids per image ID
        df_pred = pd.DataFrame(ensemble_probs, columns=classes)
        df_pred["id"] = ids

        # Group by ID and compute mean to get final prediction per image
        df_agg = df_pred.groupby("id").mean().reset_index()

        return df_agg
