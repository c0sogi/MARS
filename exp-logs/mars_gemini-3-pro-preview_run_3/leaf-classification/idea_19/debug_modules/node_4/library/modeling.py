import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import (
    save_pickle,
    load_pickle,
    save_submission,
    clip_probabilities,
    ensure_directory,
    save_numpy,
    load_numpy,
)
from library.manifold_densification import (
    get_densified_train_data,
    get_densified_test_data,
)


class DualStreamLDA(BaseEstimator, ClassifierMixin):
    """
    Implements the Hyper-Densified Dual-Stream LDA architecture.
    Wraps a Scikit-Learn pipeline that handles feature splitting, independent PCA,
    global Gaussianization, and LDA classification.
    """

    def __init__(self, dino_dim=1024, pca_variance=0.99):
        self.dino_dim = dino_dim
        self.pca_variance = pca_variance
        self.pipeline = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the pipeline.
        X is expected to be concatenated [DINO, ConvNeXt, Tabular].
        """
        # Determine dimensions
        total_dim = X.shape[1]
        # Assuming X is [DINO (1024) | ConvNeXt (1536) | Tabular (192)]
        # We calculate the split points
        # DINO ends at dino_dim
        # ConvNeXt ends at total_dim - tabular_dim.
        # However, we can just treat the visual part vs tabular part if needed,
        # but the prompt specifies independent PCA for visual streams.

        # We assume the standard extraction sizes:
        # DINO: 0 to dino_dim
        # ConvNeXt: dino_dim to (dino_dim + 1536)
        # Tabular: remainder

        # To make it robust, we define the slices based on the known DINO dimension
        # and assume the rest of the visual part is ConvNeXt.
        # The tabular features are appended last.
        # Since we don't know exact tabular dim inside the class without passing it,
        # we will assume the input X is constructed correctly.

        # Slice 1: DINO Features
        dino_slice = slice(0, self.dino_dim)

        # Slice 2: ConvNeXt Features (from DINO end to start of Tabular)
        # We know Tabular is 192, but let's rely on the fact that we apply PCA to the visual parts.
        # Let's assume the last 192 are tabular? No, safer to rely on column indices if we can.
        # Given the fixed architecture:
        # Visual = 2560. Tabular = 192. Total = 2752.
        visual_dim = 2560

        convnext_slice = slice(self.dino_dim, visual_dim)
        tabular_slice = slice(visual_dim, total_dim)

        # Define the ColumnTransformer for Independent Subspace Reduction
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "dino_pca",
                    PCA(n_components=self.pca_variance, whiten=False),
                    dino_slice,
                ),
                (
                    "conv_pca",
                    PCA(n_components=self.pca_variance, whiten=False),
                    convnext_slice,
                ),
                ("tab_pass", "passthrough", tabular_slice),
            ],
            verbose_feature_names_out=False,
        )

        # Define the Full Pipeline
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("gaussianizer", QuantileTransformer(output_distribution="normal")),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        )

        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)


def train_and_evaluate(load_cached_data=True):
    """
    Executes the Stratified K-Fold training loop with Manifold Densification.
    Saves models and class metadata.
    """
    print("Loading densified training data...")
    # X_img: (N*9, 2560), X_tab: (N*9, 192), y: (N*9,), ids: (N*9,)
    X_img, X_tab, y_raw, ids = get_densified_train_data(
        load_cached_data=load_cached_data
    )

    # Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    classes = le.classes_

    # Save classes for inference
    save_numpy(classes, Config.CACHE_CLASSES)
    print(f"Classes saved: {len(classes)} unique species.")

    # Prepare for K-Fold
    # We must split based on unique Image IDs to prevent data leakage
    unique_ids = np.unique(ids)

    # We need the label for each unique ID to stratify correctly.
    # We can pick the label from the first occurrence of each ID.
    unique_indices = []
    for uid in unique_ids:
        # Find first index where this ID appears
        idx = np.where(ids == uid)[0][0]
        unique_indices.append(idx)

    unique_labels = y_encoded[unique_indices]

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx_unique, val_idx_unique) in enumerate(
        skf.split(unique_ids, unique_labels)
    ):
        # Map unique ID indices back to actual IDs
        train_ids_fold = unique_ids[train_idx_unique]
        val_ids_fold = unique_ids[val_idx_unique]

        # Create masks for the densified dataset
        train_mask = np.isin(ids, train_ids_fold)
        val_mask = np.isin(ids, val_ids_fold)

        # Split Data
        X_train_img = X_img[train_mask]
        X_train_tab = X_tab[train_mask]
        y_train = y_encoded[train_mask]

        X_val_img = X_img[val_mask]
        X_val_tab = X_tab[val_mask]
        y_val = y_encoded[val_mask]
        val_ids_expanded = ids[val_mask]

        # Concatenate features for the pipeline
        X_train_full = np.hstack([X_train_img, X_train_tab])
        X_val_full = np.hstack([X_val_img, X_val_tab])

        # Initialize and Train Model
        model = DualStreamLDA(dino_dim=1024, pca_variance=Config.PCA_VARIANCE)
        model.fit(X_train_full, y_train)

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        save_pickle(model, model_path)

        # Evaluation: Full-Manifold Test-Time Aggregation
        # 1. Predict on all centroids
        probs_expanded = model.predict_proba(X_val_full)

        # 2. Aggregate by ID (Mean of 9 centroids)
        # We create a DataFrame to handle the grouping easily
        df_val_preds = pd.DataFrame(probs_expanded, columns=classes)
        df_val_preds["id"] = val_ids_expanded

        # Group by ID and compute mean probability vector
        df_agg = df_val_preds.groupby("id").mean()

        # Get true labels for these IDs (order must match df_agg)
        # df_agg index is the ID. We need to map these IDs to their true labels.
        # We can create a mapping from ID to Label
        id_to_label_map = dict(zip(unique_ids, unique_labels))
        y_val_true = np.array([id_to_label_map[uid] for uid in df_agg.index])

        # Compute Log Loss
        # Clip probabilities for stability
        probs_agg = clip_probabilities(df_agg.values)
        score = log_loss(y_val_true, probs_agg, labels=range(len(classes)))
        fold_scores.append(score)

        print(f"Fold {fold} | Val Log Loss: {score:.6f}")

    print(
        f"\nAverage Log Loss: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})"
    )


def generate_submission(load_cached_data=True):
    """
    Generates the final submission using the ensemble of trained models
    and Full-Manifold Test-Time Aggregation on the test set.
    """
    print("Loading densified test data...")
    # X_test_img: (N, 9, 2560), X_test_tab: (N, 9, 192), test_ids: (N,)
    X_test_img, X_test_tab, test_ids = get_densified_test_data(
        load_cached_data=load_cached_data
    )

    # Load Classes
    classes = load_numpy(Config.CACHE_CLASSES)

    # Prepare Data for Inference
    # Flatten structure: (N, 9, D) -> (N*9, D)
    N, C, D_img = X_test_img.shape
    _, _, D_tab = X_test_tab.shape

    X_test_img_flat = X_test_img.reshape(N * C, D_img)
    X_test_tab_flat = X_test_tab.reshape(N * C, D_tab)
    X_test_full = np.hstack([X_test_img_flat, X_test_tab_flat])

    # Accumulate predictions from all folds
    ensemble_probs = np.zeros((N * C, len(classes)))

    print(f"Inference with {Config.N_FOLDS} models...")

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        model = load_pickle(model_path)

        # Predict on all centroids
        probs = model.predict_proba(X_test_full)
        ensemble_probs += probs

    # Average over models
    ensemble_probs /= Config.N_FOLDS

    # Reshape back to (N, 9, n_classes) to aggregate centroids
    probs_structured = ensemble_probs.reshape(N, C, len(classes))

    # Average over centroids (Full-Manifold Aggregation)
    final_probs = probs_structured.mean(axis=1)

    # Clip probabilities
    final_probs = clip_probabilities(final_probs)

    # Create Submission DataFrame
    df_sub = pd.DataFrame(final_probs, columns=classes)
    df_sub.insert(0, "id", test_ids)

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(df_sub)
    print("Submission generated successfully.")
