import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.data_manager import DataManager


class LDAPipeline:
    """
    A wrapper class for the Leaf Classification pipeline involving:
    1. StandardScaler: Normalization.
    2. PCA: Dimensionality reduction retaining 99% variance.
    3. LDA: Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
    """

    def __init__(self):
        """
        Initializes the pipeline with configuration parameters.
        """
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=Config.PCA_VARIANCE,
                        svd_solver="full",
                        random_state=Config.SEED,
                    ),
                ),
                (
                    "lda",
                    LinearDiscriminantAnalysis(
                        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the pipeline to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.

        Returns:
            self
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Matrix of class probabilities (N_samples, N_classes).
        """
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self):
        """
        Returns the class labels known to the classifier.
        """
        return self.pipeline.named_steps["lda"].classes_


def run_pipeline(load_cached_features: bool = True):
    """
    Orchestrates the full Manifold-Densified LDA pipeline:
    1. Loads metadata.
    2. Extracts/Loads features via DataManager.
    3. Densifies training data (3x augmentation).
    4. Trains the model.
    5. Evaluates on validation set.
    6. Generates submission for test set.

    Args:
        load_cached_features (bool): If True, attempts to load features from disk.
    """
    seed_everything(Config.SEED)

    print("Initializing Pipeline Execution...")

    # 1. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    print(
        f"Loaded metadata: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
    )

    # 2. Initialize Data Manager
    dm = DataManager()

    # 3. Extract/Load Image Features (Heavy Compute)
    # The DataManager handles caching logic internally
    print("\n--- Feature Extraction ---")
    train_img_feats = dm.extract_all_views(
        df_train, "train", load_cached_data=load_cached_features
    )
    val_img_feats = dm.extract_all_views(
        df_val, "val", load_cached_data=load_cached_features
    )
    test_img_feats = dm.extract_all_views(
        df_test, "test", load_cached_data=load_cached_features
    )

    # 4. Process Tabular Features (Quantile Transform)
    print("\n--- Tabular Processing ---")
    train_tab_feats, val_tab_feats, test_tab_feats = dm.process_tabular_features(
        df_train, df_val, df_test
    )

    # 5. Prepare Data Topology
    print("\n--- Data Topology Preparation ---")

    # A. Training: Centroid Aggregation (1 Centroid per sample)
    # We need labels for training
    y_train_raw = df_train["species"].values
    X_train_img_agg, X_train_tab_agg, y_train_agg = dm.aggregate_training_data(
        train_img_feats, train_tab_feats, y_train_raw
    )

    # Fuse aggregated features
    X_train_fused = dm.fuse_features(X_train_img_agg, X_train_tab_agg)
    print(f"Final Training Set Shape: {X_train_fused.shape}")

    # B. Validation: Standard Inference Topology (1 Centroid per sample)
    X_val_img_centroid, X_val_tab = dm.prepare_inference_data(
        val_img_feats, val_tab_feats
    )
    X_val_fused = dm.fuse_features(X_val_img_centroid, X_val_tab)
    y_val = df_val["species"].values

    # C. Test: Standard Inference Topology
    X_test_img_centroid, X_test_tab = dm.prepare_inference_data(
        test_img_feats, test_tab_feats
    )
    X_test_fused = dm.fuse_features(X_test_img_centroid, X_test_tab)

    # 6. Model Training
    print("\n--- Model Training ---")
    model = LDAPipeline()
    model.fit(X_train_fused, y_train_agg)
    print("Model fitted successfully.")

    # 7. Validation Evaluation
    print("\n--- Validation ---")
    val_probs = model.predict(X_val_fused)

    # Clip probabilities for log loss stability
    val_probs_clipped = clip_probabilities(val_probs)

    # Calculate metric
    score = log_loss(y_val, val_probs_clipped, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {score}")

    # 8. Submission Generation
    print("\n--- Generating Submission ---")
    test_probs = model.predict(X_test_fused)
    test_probs_clipped = clip_probabilities(test_probs)

    # Create submission DataFrame
    # Columns must be: id, Species1, Species2, ...
    submission_df = pd.DataFrame(test_probs_clipped, columns=model.classes_)
    submission_df.insert(0, "id", df_test["id"].values)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
