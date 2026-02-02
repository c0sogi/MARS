import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import FLOAT_PRECISION, RANDOM_SEED, LDA_SOLVER


class Float64Transformer(BaseEstimator, TransformerMixin):
    """
    Transformer that casts the input data to strict float64 precision
    to minimize numerical noise at the metric floor.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to numpy array with specified precision
        return np.array(X, dtype=FLOAT_PRECISION)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from a pandas DataFrame.
    Used to decompose the feature space into Margin, Shape, Texture, etc.
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is a DataFrame for column selection
        if isinstance(X, pd.DataFrame):
            # Return values as float64
            return X[self.feature_names].values.astype(FLOAT_PRECISION)
        else:
            raise TypeError("FeatureSelector expects a pandas DataFrame as input.")


def make_alignment_pipeline(random_state=RANDOM_SEED):
    """
    Creates the Rotational Alignment pipeline:
    PowerTransformer -> PCA(whiten=False) -> PowerTransformer

    Aligns data with principal axes without noise amplification.
    """
    return Pipeline(
        [
            ("float64", Float64Transformer()),
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(whiten=False, random_state=random_state)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
        ]
    )


def make_bottleneck_pipeline(n_components=5, random_state=RANDOM_SEED):
    """
    Creates the Discriminative Bottleneck pipeline:
    Alignment Pipeline -> LDA Projection

    Projects aligned features onto the top k class-discriminative axes.
    Note: Requires 'y' during fit.
    """
    return Pipeline(
        [
            ("alignment", make_alignment_pipeline(random_state)),
            (
                "lda_proj",
                LinearDiscriminantAnalysis(n_components=n_components, solver="svd"),
            ),
        ]
    )


def make_polynomial_pipeline(degree=2, random_state=RANDOM_SEED):
    """
    Creates the Polynomial pipeline:
    PowerTransformer -> PolynomialFeatures -> PowerTransformer

    Captures non-linear physical constraints.
    """
    return Pipeline(
        [
            ("float64", Float64Transformer()),
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
        ]
    )


def make_robust_pipeline(
    n_quantiles=50, output_distribution="normal", random_state=RANDOM_SEED
):
    """
    Creates the Robust pipeline:
    QuantileTransformer

    Strictly constrained rank-based normalization to handle skew.
    """
    return Pipeline(
        [
            ("float64", Float64Transformer()),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution=output_distribution,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_marginal_pipeline():
    """
    Creates the Marginal pipeline:
    PowerTransformer

    Stabilizes variance feature-wise.
    """
    return Pipeline(
        [
            ("float64", Float64Transformer()),
            ("pt", PowerTransformer(method="yeo-johnson")),
        ]
    )
