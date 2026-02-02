import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.utils import set_seed

# Define feature dimensions based on the specific models and metadata used
# High-Res: sentence-transformers/all-MiniLM-L6-v2 -> 384 dim
# Low-Res: sentence-transformers/all-mpnet-base-v2 -> 768 dim
# Metadata: 10 numerical features defined in library.data_loader
DIM_HIGH_RES = 384
DIM_LOW_RES = 768
DIM_METADATA = 10


def combine_features(emb_high, emb_low, metadata):
    """
    Concatenates the three feature views into a single matrix for the pipeline.

    Args:
        emb_high (np.ndarray): High-resolution embeddings (N, 384).
        emb_low (np.ndarray): Low-resolution embeddings (N, 768).
        metadata (np.ndarray): Numerical metadata (N, 10).

    Returns:
        np.ndarray: Combined feature matrix (N, 1162).
    """
    # Ensure inputs are 2D
    if emb_high.ndim == 1:
        emb_high = emb_high.reshape(1, -1)
    if emb_low.ndim == 1:
        emb_low = emb_low.reshape(1, -1)
    if metadata.ndim == 1:
        metadata = metadata.reshape(1, -1)

    return np.hstack([emb_high, emb_low, metadata])


def build_adrsf_pipeline(
    pca_components=32,
    n_estimators=20,
    C=1.0,
    class_weight="balanced",
    random_state=42,
):
    """
    Constructs the Asymmetric Dual-Resolution Semantic Fusion (ADRSF) pipeline.

    Structure:
    1. Preprocessing (ColumnTransformer):
       - View 1 (High-Res, cols 0-384): L2 Normalization.
       - View 2 (Low-Res, cols 384-1152): PCA (32 dims) -> L2 Normalization.
       - View 3 (Metadata, cols 1152-1162): QuantileTransformer (Normal distribution).
    2. Classifier:
       - BaggingClassifier wrapping LogisticRegression (Ridge).

    Args:
        pca_components (int): Number of components for PCA on auxiliary view.
        n_estimators (int): Number of estimators in the Bagging ensemble.
        C (float): Inverse of regularization strength for Logistic Regression.
        class_weight (str or dict): Class weight strategy (e.g., 'balanced').
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    set_seed(random_state)

    # Define column slices for the ColumnTransformer
    # Note: slice objects work efficiently with numpy arrays in ColumnTransformer
    slice_high = slice(0, DIM_HIGH_RES)
    slice_low = slice(DIM_HIGH_RES, DIM_HIGH_RES + DIM_LOW_RES)
    slice_meta = slice(
        DIM_HIGH_RES + DIM_LOW_RES, DIM_HIGH_RES + DIM_LOW_RES + DIM_METADATA
    )

    # --- 1. Define Transformers for each View ---

    # View 1: Primary Semantics (MiniLM)
    # Strategy: L2 Normalize to project onto hypersphere
    transformer_high = Normalizer(norm="l2")

    # View 2: Auxiliary Semantics (MPNet)
    # Strategy: Asymmetric Dimensionality Reduction (PCA) -> L2 Normalize
    # We normalize AFTER PCA to ensure these features have unit norm scale like View 1
    transformer_low = Pipeline(
        steps=[
            ("pca", PCA(n_components=pca_components, random_state=random_state)),
            ("norm", Normalizer(norm="l2")),
        ]
    )

    # View 3: Robust Metadata
    # Strategy: RankGauss (QuantileTransformer) to neutralize outliers and align distribution
    transformer_meta = QuantileTransformer(
        output_distribution="normal", random_state=random_state
    )

    # Combine into Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("view_high", transformer_high, slice_high),
            ("view_low", transformer_low, slice_low),
            ("view_meta", transformer_meta, slice_meta),
        ],
        n_jobs=-1,
    )

    # --- 2. Define Classifier ---

    # Base Estimator: Logistic Regression
    # High-bias linear core, using L2 (Ridge) penalty
    base_lr = LogisticRegression(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    # Ensemble: Bagging
    # Reduces variance of the linear estimator
    bagging_clf = BaggingClassifier(
        estimator=base_lr,
        n_estimators=n_estimators,
        max_samples=1.0,  # Standard bootstrapping
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
    )

    # --- 3. Assemble Pipeline ---
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", bagging_clf)]
    )

    return pipeline
