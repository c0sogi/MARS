import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects a specific range of columns from a numpy array.
    Used to split the combined feature matrix into Global View and Morphometric View.
    """

    def __init__(self, start_idx: int, end_idx: int = None):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expecting X to be a numpy array (float64)
        if self.end_idx is None:
            return X[:, self.start_idx :]
        return X[:, self.start_idx : self.end_idx]


def get_lda_estimator(solver="lsqr", shrinkage=None, covariance_estimator=None):
    """
    Constructs a LinearDiscriminantAnalysis estimator.

    Args:
        solver (str): 'lsqr', 'eigen', or 'svd'.
        shrinkage (float, str, or None): Shrinkage parameter (e.g., 'auto', 0.01).
        covariance_estimator (object): An instance of a covariance estimator (e.g., OAS()).
                                     If provided, solver must be 'lsqr' or 'eigen'.
    """
    if covariance_estimator is not None:
        # Use specific covariance estimator (e.g., OAS)
        # Note: shrinkage parameter is typically ignored if covariance_estimator is provided
        return LinearDiscriminantAnalysis(
            solver="lsqr", covariance_estimator=covariance_estimator
        )
    else:
        # Use standard shrinkage (Fixed or Ledoit-Wolf 'auto')
        return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)


def get_topology_a(solver="lsqr", shrinkage=None, covariance_estimator=None):
    """
    Topology A: Marginal Statistical Anchors.

    Pipeline:
    1. Select Global Features (0-192)
    2. PowerTransformer (Yeo-Johnson)
    3. LDA Classifier (Robust Covariance)
    """
    return Pipeline(
        [
            ("selector", ColumnSelector(0, 192)),
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("clf", get_lda_estimator(solver, shrinkage, covariance_estimator)),
        ]
    )


def get_topology_b(solver="lsqr", shrinkage=None, covariance_estimator=None):
    """
    Topology B: Rotational Statistical Experts.

    Pipeline:
    1. Select Global Features (0-192)
    2. PowerTransformer
    3. PCA (Alignment, no whitening)
    4. PowerTransformer
    5. LDA Classifier
    """
    return Pipeline(
        [
            ("selector", ColumnSelector(0, 192)),
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(whiten=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", get_lda_estimator(solver, shrinkage, covariance_estimator)),
        ]
    )


def get_topology_c(solver="lsqr", shrinkage=None, covariance_estimator=None):
    """
    Topology C: Discriminative-Interaction Experts.

    Pipeline:
    1. Select Global Features (0-192)
    2. PowerTransformer
    3. LDA Projection (Transformer) -> 25 Components
    4. Polynomial Expansion (Degree 2)
    5. PowerTransformer
    6. LDA Classifier
    """
    # Projector projects data onto the most discriminative axes
    # We use 'svd' for the projection step because 'lsqr' does not support transform.
    # 'svd' does not support shrinkage, so we remove that parameter.
    projector = LinearDiscriminantAnalysis(n_components=25, solver="svd")

    return Pipeline(
        [
            ("selector", ColumnSelector(0, 192)),
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("lda_proj", projector),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", get_lda_estimator(solver, shrinkage, covariance_estimator)),
        ]
    )


def get_topology_d(solver="lsqr", shrinkage="auto", covariance_estimator=None):
    """
    Topology D: Polynomial Physical Experts.

    Pipeline:
    1. Select Morphometric Features (192+)
    2. PowerTransformer
    3. Polynomial Expansion (Degree 2)
    4. PowerTransformer
    5. LDA Classifier (Default: Ledoit-Wolf)
    """
    return Pipeline(
        [
            ("selector", ColumnSelector(192, None)),
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", get_lda_estimator(solver, shrinkage, covariance_estimator)),
        ]
    )
