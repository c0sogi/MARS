import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from library.config import FLOAT_PRECISION, RANDOM_SEED


class MarginalTopology(BaseEstimator, TransformerMixin):
    """
    Topology A: Marginal Parametric Anchors (The Baseline).

    Mechanism:
        Gaussianizes each feature independently using the Yeo-Johnson Power Transform.

    Pipeline:
        1. PowerTransformer(method='yeo-johnson', standardize=True)
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [("pt", PowerTransformer(method="yeo-johnson", standardize=True))]
        )

    def fit(self, X, y=None):
        """
        Fits the PowerTransformer to the data.
        """
        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """
        Applies the Power Transform.
        """
        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        X_trans = self.pipeline.transform(X)
        return X_trans.astype(FLOAT_PRECISION)


class SpectralTopology(BaseEstimator, TransformerMixin):
    """
    Topology B: Spectral Parametric Experts (The Innovation).

    Mechanism:
        First decorrelates the data and scales to unit variance (Whitening PCA),
        then applies the Power Transform to the Principal Components. This better
        approximates Multivariate Normality than marginal methods.

    Pipeline:
        1. StandardScaler
        2. PCA(whiten=True)
        3. PowerTransformer(method='yeo-johnson', standardize=True)
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(whiten=True, random_state=RANDOM_SEED)),
                ("pt", PowerTransformer(method="yeo-johnson", standardize=True)),
            ]
        )

    def fit(self, X, y=None):
        """
        Fits the Spectral pipeline (Scaling -> PCA -> PT).
        """
        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """
        Applies the Spectral transformation.
        """
        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        X_trans = self.pipeline.transform(X)
        return X_trans.astype(FLOAT_PRECISION)


class RankTopology(BaseEstimator, TransformerMixin):
    """
    Topology C: Constrained Non-Parametric Experts.

    Mechanism:
        Strictly constrained rank-based normalization using QuantileTransformer.
        Handles skewed distributions that Yeo-Johnson cannot fit.

    Pipeline:
        1. QuantileTransformer(output_distribution='normal', n_quantiles=50)
    """

    def __init__(self, n_quantiles=50):
        self.n_quantiles = n_quantiles
        self.pipeline = Pipeline(
            [
                (
                    "qt",
                    QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=self.n_quantiles,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        )

    def fit(self, X, y=None):
        """
        Fits the QuantileTransformer.
        """
        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """
        Applies the Rank-based Gaussianization.
        """
        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        X_trans = self.pipeline.transform(X)
        return X_trans.astype(FLOAT_PRECISION)
