import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config


def create_pipeline(
    feature_slices,
    pca_components=Config.AUX_PCA_COMPONENTS,
    n_bagging_estimators=Config.N_BAGGING_ESTIMATORS,
    lr_C=1.0,
    lr_class_weight=None,
    lr_max_iter=Config.LR_MAX_ITER,
    random_state=Config.SEED,
):
    """
    Constructs the Asymmetric Dual-Backbone Consensus (ADBC) pipeline.

    Args:
        feature_slices (dict): Dictionary mapping view names ('primary', 'aux', 'meta')
                               to slice objects defining column ranges in the input matrix.
        pca_components (int): Number of components for PCA on the auxiliary view.
        n_bagging_estimators (int): Number of base estimators in the Bagging ensemble.
        lr_C (float): Inverse of regularization strength for Logistic Regression.
        lr_class_weight (str or dict): Class weights for Logistic Regression (e.g., 'balanced').
        lr_max_iter (int): Maximum iterations for the Logistic Regression solver.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed training pipeline.
    """

    # 1. Define Preprocessing Steps for Each View

    # Primary View: High-Resolution Anchor (MiniLM)
    # Strategy: L2 Normalization to project onto hypersphere
    primary_transformer = Normalizer(norm="l2")

    # Auxiliary View: Low-Resolution World Knowledge (MPNet)
    # Strategy: Asymmetric Dimensionality Reduction (PCA) -> L2 Normalization
    # Note: Normalization happens AFTER PCA to ensure unit-norm scale matches primary view
    aux_transformer = Pipeline(
        [
            ("pca", PCA(n_components=pca_components, random_state=random_state)),
            ("norm", Normalizer(norm="l2")),
        ]
    )

    # Metadata View: Numerical Features
    # Strategy: RankGauss (QuantileTransformer) to normalize distributions and handle outliers
    meta_transformer = QuantileTransformer(
        output_distribution="normal", random_state=random_state
    )

    # 2. Assemble Feature Preprocessor
    # ColumnTransformer applies specific transformers to specific column slices
    preprocessor = ColumnTransformer(
        transformers=[
            ("primary", primary_transformer, feature_slices["primary"]),
            ("aux", aux_transformer, feature_slices["aux"]),
            ("meta", meta_transformer, feature_slices["meta"]),
        ],
        remainder="drop",  # Drop any columns not explicitly sliced (safety check)
    )

    # 3. Define Classifier
    # Strategy: Bagged Ensemble of Logistic Regression Classifiers
    # We use Ridge (L2) regularization by default as per strategy
    base_estimator = LogisticRegression(
        penalty=Config.LR_PENALTY,
        solver=Config.LR_SOLVER,
        C=lr_C,
        class_weight=lr_class_weight,
        max_iter=lr_max_iter,
        random_state=random_state,
    )

    classifier = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_bagging_estimators,
        max_samples=1.0,  # Use full dataset size for bootstrapping
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,  # Parallelize bagging
    )

    # 4. Construct Final Pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline
