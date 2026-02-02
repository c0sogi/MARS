import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import Normalizer, QuantileTransformer
from library import config
from library.custom_transformers import ArraySelector, WhitenedPCANormalizer


def create_model_pipeline(meta_dim, title_dim=384, body_dim=384, global_dim=768):
    """
    Constructs the W-MF-ADBE (Whitened Multi-Field Asymmetric Dual-Backbone Ensemble) pipeline.

    The pipeline splits a concatenated feature matrix into four views, processes them
    independently (including Whitened PCA for the global view), and feeds the fused
    representation into a Bagged Logistic Regression ensemble.

    Args:
        meta_dim (int): Number of metadata features (last segment of input).
        title_dim (int): Dimension of title embeddings (default 384 for MiniLM).
        body_dim (int): Dimension of body embeddings (default 384 for MiniLM).
        global_dim (int): Dimension of global embeddings (default 768 for MPNet).

    Returns:
        sklearn.pipeline.Pipeline: The constructed Scikit-Learn pipeline.
    """

    # Define indices for slicing the concatenated input array
    # Input structure: [Title (384) | Body (384) | Global (768) | Metadata (N)]
    idx_title_start = 0
    idx_title_end = title_dim

    idx_body_start = idx_title_end
    idx_body_end = idx_body_start + body_dim

    idx_global_start = idx_body_end
    idx_global_end = idx_global_start + global_dim

    idx_meta_start = idx_global_end
    # Metadata goes until the end of the array

    # 1. View 1: Title Semantics (High-Resolution Anchor)
    # L2 Normalization projects embeddings onto the hypersphere
    title_pipeline = Pipeline(
        [
            (
                "selector",
                ArraySelector(start_index=idx_title_start, end_index=idx_title_end),
            ),
            ("normalizer", Normalizer(norm="l2")),
        ]
    )

    # 2. View 2: Body Semantics (High-Resolution Anchor)
    # L2 Normalization projects embeddings onto the hypersphere
    body_pipeline = Pipeline(
        [
            (
                "selector",
                ArraySelector(start_index=idx_body_start, end_index=idx_body_end),
            ),
            ("normalizer", Normalizer(norm="l2")),
        ]
    )

    # 3. View 3: Global Context (Whitened Auxiliary)
    # Whitening normalizes variance of components to aid the linear solver
    # L2 Normalization is applied *after* whitening by the custom transformer
    global_pipeline = Pipeline(
        [
            (
                "selector",
                ArraySelector(start_index=idx_global_start, end_index=idx_global_end),
            ),
            ("whitened_pca", WhitenedPCANormalizer(n_components=config.PCA_COMPONENTS)),
        ]
    )

    # 4. View 4: Robust Metadata
    # RankGauss (QuantileTransformer) neutralizes outliers and enforces normality
    meta_pipeline = Pipeline(
        [
            ("selector", ArraySelector(start_index=idx_meta_start)),  # Selects to end
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal", random_state=config.SEED
                ),
            ),
        ]
    )

    # Feature Fusion
    # Combines all processed views into a single feature vector
    preprocessor = FeatureUnion(
        [
            ("title_view", title_pipeline),
            ("body_view", body_pipeline),
            ("global_view", global_pipeline),
            ("meta_view", meta_pipeline),
        ]
    )

    # Classifier Definition
    # Base Estimator: Logistic Regression (Ridge)
    # Note: Hyperparameters like C and class_weight will be tuned via GridSearch
    base_estimator = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=1000, random_state=config.SEED
    )

    # Ensemble: Bagging
    # Reduces variance of the linear estimator
    classifier = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=config.N_ESTIMATORS,
        random_state=config.SEED,
        n_jobs=-1,  # Parallelize bagging
    )

    # Assemble final pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline
